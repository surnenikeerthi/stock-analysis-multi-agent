from langgraph.graph import StateGraph, START, MessagesState, END
from typing import TypedDict,Any, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
import llm
import json
from langgraph.types import interrupt

import llm_chains
import sqlite3

DB_PATH = "stock.db"
required_fields=['stock_name',
                   'ticker',
                   'bought_price',
                   'stock_count',
                   'total_amount']


class PortfolioManagerAgent(MessagesState):
    context : Any

def ask_more_info(state: Annotated[PortfolioManagerAgent, InjectedState]):
    llm.llm_normal.invoke("check if more info is needed"
                          "ask the required fields and get info via tools")



#@tool
def get_more_info(query: str):
    """Ask the user for additional information needed to complete the stock operation."""

    # For now, just return what information is missing.
    # Later you can connect this to your human-in-the-loop flow.
    return f"More information may be required for this request: {query}"


@tool
def get_stock_details(
    state: Annotated[PortfolioManagerAgent, InjectedState]
):
    """Get the details of a stock already stored in the database."""
    ticker = state["context"]["ticker"]
    bought_price = state['context']['bought_price']


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
            WHERE stock_ticker = ?
            """,
            (ticker)
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
    state: Annotated[PortfolioManagerAgent, InjectedState]
):
    """Add a new stock investment to the database."""

    context = state["context"]
    stock_name = context["stock_name"]
    ticker = context["ticker"]
    bought_price = context["bought_price"]
    stock_count = context["stock_count"]
    total_amount = context["total_amount"]

    conn = sqlite3.connect(DB_PATH)
    missing_fields = [field for field in required_fields if state["context"][field] is None]
    if len(missing_fields) > 0:
        print("yyyyyyyyyyyyyyyyyyyyyyy")
        question = "please provide the required details:"+("\n").join(missing_fields)
        print("heyy")
        resp = interrupt(question)
        print("interupt caleld ")
        prompt_2 = f'''from the given statement return the asked from the question : {question}, this is the response: {resp}, so from that give the asked details in json format
 do not have json in response atall , 

'''
        chain = llm_chains.generatellmchain_withparser(llm.llm_normal, prompt_2)
        response = chain.invoke(prompt_2)
        print(response)
        #no need of json.loads() because of the parser added in chain , it automatically returns a dict value, where as json.loads() is used for string
        response_json = json.loads(response)
        print("jjjj")
        print(response_json)
    #HITL invokation

    try:
        cursor = conn.cursor()

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

        return f"Successfully added {stock_name} ({ticker}) to the portfolio."

    finally:
        conn.close()


@tool
def remove_stock_entry(
    state: Annotated[PortfolioManagerAgent, InjectedState]
):
    """Remove a stock investment from the database."""

    ticker = state["context"]["ticker"]

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM STOCKS
            WHERE stock_ticker = ?
            """,
            (ticker,)
        )

        conn.commit()

        if cursor.rowcount == 0:
            return f"No investment found for {ticker}"

        return f"Successfully removed {ticker} from the portfolio."

    finally:
        conn.close()


@tool
def update_stock_details(
    state: Annotated[PortfolioManagerAgent, InjectedState]
):
    """Update the details of an existing stock investment."""

    context = state["context"]

    stock_name = context["stock_name"]
    ticker = context["ticker"]
    bought_price = context["bought_price"]
    stock_count = context["stock_count"]
    total_amount = context["total_amount"]

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

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

        conn.commit()

        if cursor.rowcount == 0:
            return f"No investment found for {ticker}"

        return f"Successfully updated {ticker}."

    finally:
        conn.close()


@tool
def get_price_you_invested(
    state: Annotated[PortfolioManagerAgent, InjectedState]
):
    """Get the price at which the user invested in the stock."""

    ticker = state["context"]["ticker"]

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT bought_price
            FROM STOCKS
            WHERE stock_ticker = ?
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
    get_stock_details,
    update_stock_details,
    remove_stock_entry,
    add_stock_entry,
    get_price_you_invested
]

tool_node = ToolNode(tools)

def orchestrator_for_this(state : PortfolioManagerAgent):
    print("entered")
    prompt = PromptTemplate.from_template("you are only the tools deciding agent ok , here is the data i have , this is the prompt given go with the tools and perform what has to be performed {query}")
    chain = llm_chains.generatellm_with_tools(llm.llm_normal, prompt, tools)
    print(chain)
    response = chain.invoke({'query': state['context']['query']})
    print("response")
    print(response)
    return {'messages':[response]}

builder = StateGraph(PortfolioManagerAgent)
builder.add_node("ask_more_info",ask_more_info)
builder.add_node('orchestrator_for_this', orchestrator_for_this )
builder.add_node('tools', tool_node)
builder.add_edge(START, "orchestrator_for_this")
builder.add_conditional_edges("orchestrator_for_this",tools_condition)
builder.add_edge('tools', 'orchestrator_for_this')
graph = builder.compile()
#create a table