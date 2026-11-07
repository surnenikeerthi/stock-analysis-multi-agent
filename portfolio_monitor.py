from dotenv import load_dotenv
load_dotenv()

import operator
import sqlite3
from datetime import datetime, time, timedelta, timezone
from typing import Any, TypedDict, Annotated, Literal

import yfinance as yf
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from stock_analyser import agent_registry
from PortfolioManagerAgent import DB_PATH
from telegram_notifier import send_message

# ---------------------------------------------------------------------
# Automatic portfolio monitor - separate from the chat/UI flow.
# For every holding in stock.db: fetch current price, compute P&L,
# run the analysis agents, and let the LLM decide HOLD / SELL / BUY_MORE
# following the user's trading style.
# ---------------------------------------------------------------------

# below this profit %, never suggest selling (avoids alerts on small gains)
MIN_PROFIT_PCT_TO_CONSIDER_SELL = 10
# price must be at least this % below bought price to suggest buying more
BUY_MORE_MIN_DIP_PCT = 10

# fixed agent list - no LLM agent selection, no PortfolioManagerAgent
# (it can write to the DB / interrupt, which is wrong for unattended runs)
MONITOR_AGENTS = [
    "income_statement_agent",
    "balance_sheet_analysis_agent",
    "cashFlowAnalysisAgent",
    "technical_agent",
    "newsAnalysisAgent",
]

# temperature 0 so the same data gives the same decision every run
decision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class Decision(BaseModel):
    action: Literal["HOLD", "SELL", "BUY_MORE"]
    confidence: float = Field(description="Confidence in the action, between 0 and 1")
    red_flag: bool = Field(description="True only for severe company-level trouble: bankruptcy, insolvency, "
                                       "fraud, debt default, regulatory action, delisting, auditor resignation")
    reason: str = Field(description="2-3 sentences justifying the action using the supplied data")


class stock_monitor(TypedDict):
    context: Any            # holding details: stock_name, ticker, bought_price, stock_count, total_amount
    current_price: float
    pnl: dict
    full_information: str
    decision: dict


class portfolio_monitor(TypedDict):
    holdings: list
    results: Annotated[list, operator.add]   # each parallel stock run appends its result


# ---------------------------------------------------------------------
# DB / market data helpers
# ---------------------------------------------------------------------

def get_holdings_from_db():
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
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "stock_name": row[0],
            "ticker": row[1],
            "bought_price": row[2],
            "stock_count": row[3],
            "total_amount": row[4],
        }
        for row in rows
    ]


def get_current_price(ticker):
    ticker_obj = yf.Ticker(ticker)
    try:
        price = ticker_obj.fast_info["last_price"]
    except Exception:
        price = None
    if not price:
        # fall back to the latest daily close
        history = ticker_obj.history(period="5d", interval="1d")
        if history.empty:
            return None
        price = history["Close"].iloc[-1]
    return float(price)


def compute_pnl(holding, current_price):
    bought_price = holding["bought_price"]
    stock_count = holding["stock_count"] or 0
    invested = holding["total_amount"] or bought_price * stock_count
    current_value = current_price * stock_count
    return {
        "invested": round(invested, 2),
        "current_value": round(current_value, 2),
        "pnl_amount": round(current_value - invested, 2),
        "pnl_pct": round((current_price - bought_price) / bought_price * 100, 2),
    }


# ---------------------------------------------------------------------
# per-stock graph nodes
# ---------------------------------------------------------------------

def price_and_pnl(state: stock_monitor):
    holding = state["context"]
    current_price = get_current_price(holding["ticker"])
    if current_price is None:
        raise ValueError(f"No current price available for {holding['ticker']}")
    return {"current_price": current_price, "pnl": compute_pnl(holding, current_price)}


def run_monitor_agents(state: stock_monitor):
    full_information = ""
    for agent in MONITOR_AGENTS:
        # one failing agent should not stop the decision for this stock
        try:
            internalstate = agent_registry[agent].graph.invoke({"context": state["context"]})
        except Exception as e:
            print(f"[portfolio_monitor] {agent} failed for {state['context']['ticker']}: {e}")
            full_information += f"\n\n### {agent}\nNot available (agent failed)."
            continue
        if agent == "newsAnalysisAgent":
            full_information += f"\n\n### {agent}\n{internalstate['news']}"
        else:
            full_information += f"\n\n### {agent}\n{internalstate['analysis']}"
    return {"full_information": full_information}


def portfolio_decision(state: stock_monitor):
    holding = state["context"]
    pnl = state["pnl"]
    prompt = f"""You are monitoring a stock the user ALREADY OWNS and must decide: HOLD, SELL or BUY_MORE.

Holding:
- Company: {holding['stock_name']} ({holding['ticker']})
- Bought price per share: {holding['bought_price']}
- Shares: {holding['stock_count']}
- Current price: {state['current_price']}
- Invested: {pnl['invested']}, current value: {pnl['current_value']}
- Profit/Loss: {pnl['pnl_amount']} ({pnl['pnl_pct']}%)

The user's trading style - follow these rules strictly:
1. The user NEVER sells at a loss and uses NO stop-loss. A falling price alone is never a reason to sell.
2. If the position is in loss, the answer is HOLD, unless there is a RED FLAG: bankruptcy, insolvency, fraud,
   debt default, serious regulatory action, delisting, auditor resignation, or a collapsing balance sheet.
   Only then SELL, and set red_flag = true.
3. If the position is in profit of at least {MIN_PROFIT_PCT_TO_CONSIDER_SELL}%, judge whether there is still growth ahead
   using the news, the type and quality of the company (fundamentals), and whether buyers or sellers are in
   control (trend, RSI, MACD, volume). If growth still looks likely, HOLD. If momentum is fading, sellers are in
   control, or the news is negative, SELL to book profit.
4. If the profit is below {MIN_PROFIT_PCT_TO_CONSIDER_SELL}%, do not SELL unless there is a red flag.
5. BUY_MORE only if the price is at least {BUY_MORE_MIN_DIP_PCT}% below the bought price, there is no red flag,
   the company fundamentals are healthy, and the technicals look oversold or show a reversal.
6. Base every statement strictly on the data below. Do not invent numbers.
7. Missing or unavailable data (None, "not available", agent failed) is UNKNOWN, not a negative signal.
   Never use missing data as a reason to SELL - decide only on the data that is present.

Information gathered from analysis agents:
{state['full_information']}
"""
    decision = decision_llm.with_structured_output(Decision).invoke(prompt)
    return {"decision": enforce_trading_rules(decision.model_dump(), pnl["pnl_pct"])}


def enforce_trading_rules(decision, pnl_pct):
    # hard guard in code - the LLM must not override the user's style
    if decision["red_flag"]:
        return decision
    if decision["action"] == "SELL" and pnl_pct < MIN_PROFIT_PCT_TO_CONSIDER_SELL:
        decision["reason"] = f"[Overridden SELL -> HOLD: no red flag and profit below " \
                             f"{MIN_PROFIT_PCT_TO_CONSIDER_SELL}%] " + decision["reason"]
        decision["action"] = "HOLD"
    if decision["action"] == "BUY_MORE" and pnl_pct > -BUY_MORE_MIN_DIP_PCT:
        decision["reason"] = f"[Overridden BUY_MORE -> HOLD: dip smaller than " \
                             f"{BUY_MORE_MIN_DIP_PCT}%] " + decision["reason"]
        decision["action"] = "HOLD"
    return decision


stock_builder = StateGraph(stock_monitor)
stock_builder.add_node("price_and_pnl", price_and_pnl)
stock_builder.add_node("run_monitor_agents", run_monitor_agents)
stock_builder.add_node("portfolio_decision", portfolio_decision)
stock_builder.add_edge(START, "price_and_pnl")
stock_builder.add_edge("price_and_pnl", "run_monitor_agents")
stock_builder.add_edge("run_monitor_agents", "portfolio_decision")
stock_builder.add_edge("portfolio_decision", END)
monitor_graph = stock_builder.compile()


# ---------------------------------------------------------------------
# portfolio-level graph: load holdings -> one monitor_graph run per stock (parallel)
# ---------------------------------------------------------------------

def load_holdings(state: portfolio_monitor):
    return {"holdings": get_holdings_from_db()}


def fan_out_holdings(state: portfolio_monitor):
    if not state["holdings"]:
        return END
    return [Send("analyse_holding", holding) for holding in state["holdings"]]


def analyse_holding(holding: dict):
    # one bad ticker should not kill the whole run
    try:
        result = monitor_graph.invoke({"context": holding})
    except Exception as e:
        print(f"[portfolio_monitor] failed for {holding['ticker']}: {e}")
        return {"results": [{"holding": holding, "error": str(e)}]}
    return {"results": [{
        "holding": holding,
        "current_price": result["current_price"],
        "pnl": result["pnl"],
        "decision": result["decision"],
    }]}


builder = StateGraph(portfolio_monitor)
builder.add_node("load_holdings", load_holdings)
builder.add_node("analyse_holding", analyse_holding)
builder.add_edge(START, "load_holdings")
builder.add_conditional_edges("load_holdings", fan_out_holdings, ["analyse_holding", END])
builder.add_edge("analyse_holding", END)
graph = builder.compile()


def print_results(results):
    if not results:
        print("No holdings found in the database.")
        return
    for r in results:
        holding = r["holding"]
        if "error" in r:
            print(f"{holding['ticker']} | ERROR: {r['error']}")
            continue
        pnl = r["pnl"]
        decision = r["decision"]
        print(f"\n{holding['stock_name']} ({holding['ticker']})\n"
              f"  bought {holding['bought_price']:.2f} x {holding['stock_count']} | now {r['current_price']:.2f}\n"
              f"  P&L {pnl['pnl_amount']} ({pnl['pnl_pct']}%)\n"
              f"  -> {decision['action']} (confidence {decision['confidence']}, red_flag {decision['red_flag']})\n"
              f"  {decision['reason']}")


IST = timezone(timedelta(hours=5, minutes=30))
CLOSING_RUN_AFTER = time(15, 15)     # the 3:30 PM run sends the full report


def market_open_today():
    # a holiday has no Nifty 50 candle for today
    history = yf.Ticker("^NSEI").history(period="5d", interval="1d")
    if history.empty:
        return False
    return history.index[-1].date() == datetime.now(IST).date()


def needs_alert(result):
    # hourly runs only message for SELL, BUY_MORE or a red flag
    if "error" in result:
        return False
    return result["decision"]["action"] != "HOLD" or result["decision"]["red_flag"]


def format_telegram_message(results, title="📊 Portfolio check"):
    if not results:
        return f"{title}\n\nNo holdings found in the database."
    lines = [title]
    for r in results:
        holding = r["holding"]
        if "error" in r:
            lines.append(f"❗ {holding['ticker']}: analysis failed ({r['error']})")
            continue
        pnl = r["pnl"]
        decision = r["decision"]
        flag = " 🚩 RED FLAG" if decision["red_flag"] else ""
        lines.append(f"{holding['stock_name']} ({holding['ticker']})\n"
                     f"bought ₹{holding['bought_price']:,.2f} x {holding['stock_count']} | now ₹{r['current_price']:,.2f}\n"
                     f"P&L ₹{pnl['pnl_amount']:,.2f} ({pnl['pnl_pct']:+.2f}%)\n"
                     f"➜ {decision['action']}{flag} (confidence {decision['confidence']:.0%})\n"
                     f"{decision['reason']}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    now = datetime.now(IST)
    if not market_open_today():
        print(f"{now:%Y-%m-%d %H:%M} market closed today, skipping")      # holiday -> log only, no message
    else:
        output = graph.invoke({"holdings": [], "results": []})
        results = output["results"]
        print(f"\n===== run at {now:%Y-%m-%d %H:%M} =====")
        print_results(results)

        if now.time() >= CLOSING_RUN_AFTER:
            message = format_telegram_message(results)                      # 3:30 run -> full report
        else:
            alerts = [r for r in results if needs_alert(r)]
            message = format_telegram_message(alerts, "⚠️ Portfolio alert") if alerts else None   # quiet if all HOLD

        # a telegram failure should not hide the results already printed above
        if message:
            try:
                send_message(message)
                print("\nSent to Telegram.")
            except Exception as e:
                print(f"\n[portfolio_monitor] Telegram send failed: {e}")
