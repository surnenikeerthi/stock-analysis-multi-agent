from langgraph.graph import StateGraph, START, MessagesState, END
from typing import TypedDict,Any, Annotated, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
import llm
import json
import re
from datetime import datetime
from langgraph.types import interrupt, Command

import llm_chains
import sqlite3

DB_PATH = "stock.db"
required_fields=['stock_name',
                   'ticker',
                   'bought_price',
                   'stock_count',
                   'total_amount']

MAX_ASKS = 3      # how many times a question is repeated after an unusable reply
YES_WORDS = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm", "confirmed"}
CANCEL_WORDS = {"no", "n", "nope", "cancel", "stop"}


class PortfolioManagerAgent(MessagesState):
    context : Any


# ---------------------------------------------------------------------
# plain helpers - shared by the tools below, not exposed to the LLM
# ---------------------------------------------------------------------

def _holding_from_row(row):
    return {
        "stock_name": row[0],
        "ticker": row[1],
        "bought_price": row[2],
        "stock_count": row[3],
        "total_amount": row[4],
    }


def get_all_holdings():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT stock_name,
                   stock_ticker,
                   bought_price,
                   stocks_count,
                   total_amount
            FROM STOCKS
            """
        )
        return [_holding_from_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_holding(ticker):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT stock_name,
                   stock_ticker,
                   bought_price,
                   stocks_count,
                   total_amount
            FROM STOCKS
            WHERE UPPER(stock_ticker) = UPPER(?)
            """,
            (ticker,)
        )
        row = cursor.fetchone()
        return _holding_from_row(row) if row else None
    finally:
        conn.close()


def find_holding(text):
    # match a reply like "tcs", "TCS.NS" or "tata consultancy" to a holding
    text = str(text).strip().lower()
    if not text:
        return None
    for holding in get_all_holdings():
        ticker = holding["ticker"].lower()
        if text in (ticker, ticker.split(".")[0]) or text in holding["stock_name"].lower():
            return holding
    return None


def _resolve_ticker(ticker, state):
    # the LLM's argument first, the router's extracted context as fallback
    value = ticker or state["context"].get("ticker")
    return value.strip().upper() if value else None


def delete_holding(cursor, ticker):
    # shared delete step for remove, remove all and selling every share
    cursor.execute(
        """
        DELETE FROM STOCKS
        WHERE UPPER(stock_ticker) = UPPER(?)
        """,
        (ticker,)
    )
    return cursor.rowcount


def update_stock_details(cursor, stock_name, ticker, bought_price, stock_count, total_amount):
    # plain helper used by add_stock_entry, not exposed to the LLM
    cursor.execute(
        """
        UPDATE STOCKS
        SET stock_name = ?,
            bought_price = ?,
            stocks_count = ?,
            total_amount = ?
        WHERE stock_ticker = ?
        """,
        (
            stock_name,
            bought_price,
            stock_count,
            total_amount,
            ticker
        )
    )


def create_sales_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS SALES(
            sold_at TEXT,
            stock_name TEXT,
            stock_ticker TEXT,
            shares_sold REAL,
            sale_price REAL,
            avg_bought_price REAL,
            profit_booked REAL
        )
        """
    )


def sell_shares(ticker, shares, sale_price):
    # the selling rules, in one place: record the sale, then reduce or remove the holding.
    # the average bought price does not change when selling.
    holding = get_holding(ticker)
    if holding is None:
        raise ValueError(f"{ticker} is not in the portfolio.")
    owned = holding["stock_count"]
    if shares <= 0 or shares > owned:
        raise ValueError(f"You hold {owned} shares of {holding['stock_name']}, so you cannot sell {shares}.")

    avg_price = holding["bought_price"]
    profit = round((sale_price - avg_price) * shares, 2)
    remaining = owned - shares

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        create_sales_table(cursor)
        cursor.execute(
            "INSERT INTO SALES VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), holding["stock_name"], holding["ticker"],
             shares, sale_price, avg_price, profit)
        )
        if remaining == 0:
            delete_holding(cursor, holding["ticker"])
        else:
            update_stock_details(cursor, holding["stock_name"], holding["ticker"], avg_price,
                                 remaining, round(avg_price * remaining, 2))
        conn.commit()
    finally:
        conn.close()

    return {"stock_name": holding["stock_name"], "ticker": holding["ticker"], "shares_sold": shares,
            "sale_price": sale_price, "profit_booked": profit, "shares_left": remaining}


# ---------------------------------------------------------------------
# asking the user (human in the loop via interrupt)
# ---------------------------------------------------------------------

def _parse_number(text):
    # "5", "5 shares", "₹2,500", "1,50,000" -> number
    match = re.search(r"\d[\d,]*(\.\d+)?", str(text))
    return float(match.group().replace(",", "")) if match else None


def _whole(number):
    return int(number) if float(number).is_integer() else number


def _money(value):
    return f"₹{value:,.2f}"


def _is_cancel(reply):
    return reply.strip().lower().strip(".!") in CANCEL_WORDS


def ask_number(question, valid):
    # returns a valid number, or None when the user cancels or keeps giving unusable replies
    for _ in range(MAX_ASKS):
        reply = str(interrupt(question)).strip()
        if _is_cancel(reply):
            return None
        number = _parse_number(reply)
        if number is not None and valid(number):
            return number
        question = "I couldn't use that reply. " + question.split("I couldn't use that reply. ")[-1]
    return None


def ask_yes_no(question):
    reply = str(interrupt(question)).strip().lower().strip(".!")
    return reply in YES_WORDS or reply.startswith("yes")


def ask_for_holding(question):
    reply = str(interrupt(question)).strip()
    if _is_cancel(reply):
        return None
    return find_holding(reply)


def _holdings_summary():
    holdings = get_all_holdings()
    if not holdings:
        return "The portfolio is empty."
    return "Your holdings: " + ", ".join(f"{h['stock_name']} ({h['ticker']}, {_whole(h['stock_count'])} shares)"
                                         for h in holdings)


def get_more_info(state: Annotated[PortfolioManagerAgent, InjectedState], missing_fields):
    print("yyyyyyyyyyyyyyyyyyyyyyy")
    question = "please provide the required details:" + ("\n").join(missing_fields)
    print("heyy")
    resp = interrupt(question)
    print("interupt caleld ")
    prompt_2 = f'''from the given statement return the asked from the question : {question}, this is the response: {resp}, so from that give the asked details in json format
     do not have json in response atall ,

    '''
    chain = llm_chains.generatellmchain_withparser(llm.llm_normal, prompt_2)
    response = chain.invoke(prompt_2)
    print(response)
    # no need of json.loads() because of the parser added in chain , it automatically returns a dict value, where as json.loads() is used for string
    response_json = response
    print("jjjj")
    print(response_json)
    return response_json


# ---------------------------------------------------------------------
# tools the LLM can call
# ---------------------------------------------------------------------

@tool
def list_all_holdings():
    """List every stock in the user's portfolio with bought price, number of shares and total invested.
    Use this for questions like 'what stocks do I own', 'show my portfolio', or to find a stock's ticker."""
    holdings = get_all_holdings()
    if not holdings:
        return "The portfolio is empty."
    return holdings


@tool
def get_stock_details(
    state: Annotated[PortfolioManagerAgent, InjectedState],
    ticker: Annotated[Optional[str], "Yahoo Finance ticker of the stock, e.g. TCS.NS"] = None,
):
    """Get the details of one stock already stored in the database."""
    ticker = _resolve_ticker(ticker, state)
    if ticker is None:
        return "No stock was named. " + _holdings_summary()

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT stock_name,
                   stock_ticker,
                   bought_price,
                   stocks_count,
                   total_amount
            FROM STOCKS
            WHERE UPPER(stock_ticker) = UPPER(?)
            """,
            (ticker,)
        )

        result = cursor.fetchone()

        if result is None:
            return f"No investment found for {ticker}"

        return {
            "stock_name": result[0],
            "stock_ticker": result[1],
            "bought_price": result[2],
            "stocks_count": result[3],
            "total_amount": result[4],
        }

    finally:
        conn.close()


@tool
def add_stock_entry(
    state: Annotated[PortfolioManagerAgent, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    ticker: Annotated[Optional[str], "Yahoo Finance ticker, e.g. INFY.NS"] = None,
    stock_name: Annotated[Optional[str], "Company name"] = None,
    shares: Annotated[Optional[float], "Number of shares bought"] = None,
    bought_price: Annotated[Optional[float], "Price paid per share"] = None,
    total_amount: Annotated[Optional[float], "Total amount paid"] = None,
):
    """Add a stock purchase to the portfolio.
    Use this whenever the user says they bought, own or added more of a stock.
    If the stock is already in the portfolio, the new shares and amount are added to the existing
    holding; otherwise a new entry is created. Leave out any value the user did not give -
    missing details are asked from the user."""

    # arguments from the LLM first, the router's extracted context as fallback
    context = dict(state["context"])
    context["ticker"] = _resolve_ticker(ticker, state)
    context["stock_name"] = stock_name or context.get("stock_name")
    context["stock_count"] = shares if shares is not None else context.get("stock_count")
    context["bought_price"] = bought_price if bought_price is not None else context.get("bought_price")
    context["total_amount"] = total_amount if total_amount is not None else context.get("total_amount")

    _fill_derived_values(context)
    missing_fields = [field for field in required_fields if context.get(field) is None]
    if len(missing_fields) > 0:
        #HITL invokation
        new_values = get_more_info(state, missing_fields)
        for field in missing_fields:
            if(new_values.get(field) is not None):
                context[field] = new_values[field]
        _fill_derived_values(context)

    still_missing = [field for field in required_fields if context.get(field) is None]
    if still_missing:
        msg = "Nothing was added. These details are still missing: " + ", ".join(still_missing)
        return Command(update={"context": context,
                               "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

    stock_name = context["stock_name"]
    ticker = context["ticker"].strip().upper()
    bought_price = context["bought_price"]
    stock_count = context["stock_count"]
    total_amount = context["total_amount"]

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        existing = get_holding(ticker)   # looks up the ticker after the user's answers, not the old state

        if existing is not None:              # the stock is already purchased
            new_count = (existing["stock_count"] or 0) + stock_count
            new_total = (existing["total_amount"] or 0) + total_amount
            new_price = new_total / new_count
            update_stock_details(cursor, existing["stock_name"], existing["ticker"], new_price, new_count, new_total)
            conn.commit()
            msg = f"Updated {existing['stock_name']} ({existing['ticker']}): now {_whole(new_count)} shares, " \
                  f"total {_money(new_total)}, average price {_money(new_price)}."
            return Command(update={"context": context,
                                   "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

        cursor.execute(
            """
            INSERT INTO STOCKS
            (
                stock_name,
                stock_ticker,
                bought_price,
                stocks_count,
                total_amount
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                stock_name,
                ticker,
                bought_price,
                stock_count,
                total_amount
            )
        )

        conn.commit()

        msg = f"Successfully added {stock_name} ({ticker}) to the portfolio."
        return Command(update={"context": context,
                               "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

    finally:
        conn.close()


def _fill_derived_values(context):
    # any one of price / count / total can be worked out from the other two
    price, count, total = context.get("bought_price"), context.get("stock_count"), context.get("total_amount")
    if total is None and price is not None and count is not None:
        context["total_amount"] = price * count
    elif price is None and total is not None and count:
        context["bought_price"] = total / count
    elif count is None and total is not None and price:
        context["stock_count"] = round(total / price)


@tool
def sell_stock_entry(
    state: Annotated[PortfolioManagerAgent, InjectedState],
    ticker: Annotated[Optional[str], "Yahoo Finance ticker of the stock sold, e.g. TCS.NS"] = None,
    shares: Annotated[Optional[float], "Number of shares sold, only if the user said it"] = None,
    sale_price: Annotated[Optional[float], "Price per share the user sold at, only if the user said it"] = None,
    sell_all: Annotated[bool, "True only if the user said they sold all their shares"] = False,
):
    """Record that the user SOLD some or all shares of a stock they hold.
    Use this for any sale: 'I sold', 'booked profit', 'exited'. The sale and the profit booked are saved.
    Never use remove_stock_entry for a sale. Leave out shares or sale_price if the user did not say them -
    the tool asks the user."""

    ticker = _resolve_ticker(ticker, state)
    holding = get_holding(ticker) if ticker else None
    if holding is None:
        holding = ask_for_holding(("I couldn't find that stock in your portfolio. " if ticker else "")
                                  + "Which stock did you sell? " + _holdings_summary())
        if holding is None:
            return "Nothing was sold - the stock is not in the portfolio. " + _holdings_summary()

    name, owned = holding["stock_name"], holding["stock_count"]

    if sell_all:
        shares = owned
    elif shares is None:
        reply = str(interrupt(f"You hold {_whole(owned)} shares of {name}. Did you sell all {_whole(owned)}, "
                              f"or how many? Reply 'all' or a number.")).strip().lower()
        if _is_cancel(reply):
            return "Cancelled, nothing was changed."
        shares = owned if "all" in reply else _parse_number(reply)

    if shares is None or not 0 < shares <= owned:
        shares = ask_number(f"How many {name} shares did you sell? You hold {_whole(owned)}.",
                            lambda n: 0 < n <= owned)
        if shares is None:
            return "Cancelled, nothing was changed."

    if sale_price is None or sale_price <= 0:
        sale_price = ask_number(f"At what price per share did you sell {name}?", lambda n: n > 0)
        if sale_price is None:
            return "Cancelled, nothing was changed."

    if shares == owned:
        confirmed = ask_yes_no(f"Sell all {_whole(owned)} {name} shares at {_money(sale_price)}? "
                               f"This removes {name} from your portfolio. Reply yes or no.")
        if not confirmed:
            return "Cancelled, nothing was changed."

    try:
        result = sell_shares(holding["ticker"], shares, sale_price)
    except ValueError as e:
        return f"Nothing was sold: {e}"

    left = (f"You now hold {_whole(result['shares_left'])} shares." if result["shares_left"]
            else f"{name} is no longer in your portfolio.")
    return (f"Sold {_whole(shares)} {name} at {_money(sale_price)}. "
            f"Profit booked {_money(result['profit_booked'])}. {left}")


@tool
def correct_stock_entry(
    state: Annotated[PortfolioManagerAgent, InjectedState],
    ticker: Annotated[Optional[str], "Yahoo Finance ticker of the stock to correct, e.g. TCS.NS"] = None,
    shares: Annotated[Optional[float], "The correct number of shares"] = None,
    bought_price: Annotated[Optional[float], "The correct average price paid per share"] = None,
):
    """Overwrite a holding's number of shares and/or bought price to fix a wrong entry.
    Use this when the user says an entry is wrong or asks to correct / change / set a holding's values.
    Not for new purchases (add_stock_entry) or sales (sell_stock_entry)."""

    ticker = _resolve_ticker(ticker, state)
    holding = get_holding(ticker) if ticker else None
    if holding is None:
        holding = ask_for_holding("Which stock should I correct? " + _holdings_summary())
        if holding is None:
            return "Nothing was changed - the stock is not in the portfolio. " + _holdings_summary()

    if shares is None and bought_price is None:
        reply = str(interrupt(f"{holding['stock_name']} is saved as {_whole(holding['stock_count'])} shares at "
                              f"{_money(holding['bought_price'])}. What are the correct values? "
                              f"Reply like '16 at 2400'."))
        numbers = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", reply)]
        if len(numbers) < 2:
            return "Nothing was changed - I need both the number of shares and the price."
        shares, bought_price = numbers[0], numbers[1]

    new_shares = shares if shares is not None else holding["stock_count"]
    new_price = bought_price if bought_price is not None else holding["bought_price"]
    if new_shares <= 0 or new_price <= 0:
        return "Nothing was changed - shares and price must be above zero."

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        update_stock_details(cursor, holding["stock_name"], holding["ticker"], new_price,
                             new_shares, round(new_price * new_shares, 2))
        conn.commit()
    finally:
        conn.close()

    return (f"Corrected {holding['stock_name']}: {_whole(holding['stock_count'])} shares at "
            f"{_money(holding['bought_price'])} -> {_whole(new_shares)} shares at {_money(new_price)}.")


@tool
def remove_stock_entry(
    state: Annotated[PortfolioManagerAgent, InjectedState],
    ticker: Annotated[Optional[str], "Yahoo Finance ticker of the stock to delete, e.g. TCS.NS"] = None,
):
    """Delete one stock entry from the portfolio, e.g. one added by mistake. Asks the user to confirm.
    This does NOT record a sale - if the user sold shares, use sell_stock_entry instead."""

    ticker = _resolve_ticker(ticker, state)
    holding = get_holding(ticker) if ticker else None
    if holding is None:
        holding = ask_for_holding("Which stock should I remove? " + _holdings_summary())
        if holding is None:
            return "Nothing was removed - the stock is not in the portfolio. " + _holdings_summary()

    confirmed = ask_yes_no(f"Remove {holding['stock_name']} ({holding['ticker']}, {_whole(holding['stock_count'])} "
                           f"shares) from your portfolio? This deletes the entry and does not record a sale. "
                           f"Reply yes or no.")
    if not confirmed:
        return "Cancelled, nothing was removed."

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()
        delete_holding(cursor, holding["ticker"])
        conn.commit()
        return f"Successfully removed {holding['ticker']} from the portfolio."

    finally:
        conn.close()


@tool
def remove_all_holdings():
    """Delete EVERY stock from the portfolio, e.g. 'empty my portfolio' or 'remove all stocks'.
    Asks the user to confirm. Call this once - do not call remove_stock_entry repeatedly for this."""

    holdings = get_all_holdings()
    if not holdings:
        return "The portfolio is already empty."

    confirmed = ask_yes_no(f"Remove all {len(holdings)} stocks from your portfolio? {_holdings_summary()}. "
                           f"Reply yes or no.")
    if not confirmed:
        return "Cancelled, nothing was removed."

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        for holding in holdings:
            delete_holding(cursor, holding["ticker"])
        conn.commit()
    finally:
        conn.close()
    return f"Removed all {len(holdings)} stocks. The portfolio is now empty."


@tool
def get_price_you_invested(
    state: Annotated[PortfolioManagerAgent, InjectedState],
    ticker: Annotated[Optional[str], "Yahoo Finance ticker of the stock, e.g. TCS.NS"] = None,
):
    """Get the price at which the user invested in the stock."""

    ticker = _resolve_ticker(ticker, state)
    if ticker is None:
        return "No stock was named. " + _holdings_summary()

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT bought_price
            FROM STOCKS
            WHERE UPPER(stock_ticker) = UPPER(?)
            """,
            (ticker,)
        )

        result = cursor.fetchone()

        if result is None:
            return f"No investment found for {ticker}"

        return {
            "ticker": ticker,
            "bought_price": result[0]
        }

    finally:
        conn.close()
tools = [
    list_all_holdings,
    get_stock_details,
    add_stock_entry,
    sell_stock_entry,
    correct_stock_entry,
    remove_stock_entry,
    remove_all_holdings,
    get_price_you_invested
]

tool_node = ToolNode(tools)

def orchestrator_for_this(state : PortfolioManagerAgent):
    print("entered")
    context = state["context"]
    system = SystemMessage(
        "You are the tool-deciding agent for a stock portfolio. Call the tools needed for the "
        "user's request. Once a tool result shows the work is done, reply to the user with a "
        "short summary and do not call any more tools.\n"
        "- Pass the stock's Yahoo Finance ticker (Indian stocks end in .NS, e.g. TCS.NS) and the numbers "
        "from the user's message as tool arguments. Never invent a number: leave it out and the tool asks.\n"
        "- Bought / purchased -> add_stock_entry. Sold -> sell_stock_entry, never remove_stock_entry.\n"
        "- Wrong entry to fix -> correct_stock_entry. Delete one entry -> remove_stock_entry. "
        "Empty the whole portfolio -> remove_all_holdings, called once.\n"
        "- Questions about all holdings -> list_all_holdings.\n"
        "- If a tool says the user cancelled, just confirm that nothing was changed. Do not call more tools "
        "or suggest other actions.\n"
        f"Stock named in the message, if any: {context.get('stock_name')} ({context.get('ticker')})."
    )
    # one tool call at a time, so confirmation questions are asked one after another
    llm_tools = llm.llm_normal.bind_tools(tools, parallel_tool_calls=False)

    if not state["messages"]:  # first pass: start the history with the user's query
        query = HumanMessage(context["query"])
        response = llm_tools.invoke([system, query])
        print(response)
        return {"messages": [query, response]}

    response = llm_tools.invoke([system] + state["messages"])  # later passes see the tool results
    print(response)
    return {"messages": [response]}

builder = StateGraph(PortfolioManagerAgent)
builder.add_node('orchestrator_for_this', orchestrator_for_this )
builder.add_node('tools', tool_node)
builder.add_edge(START, "orchestrator_for_this")
builder.add_conditional_edges("orchestrator_for_this",tools_condition)
builder.add_edge('tools', 'orchestrator_for_this')
graph = builder.compile()
#create a table
