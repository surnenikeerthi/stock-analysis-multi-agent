from typing import TypedDict
import llm
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

import sqlite3
from datetime import datetime


class PortfolioManagerAgent(MessagesState):
    context: dict
    intent: str
    response: str


DB = "portfolio_1.db"


def get_connection():
    return sqlite3.connect(DB, check_same_thread=False)


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---- Node 1: decide store vs fetch ----
def classify_intent(state: PortfolioManagerAgent):
    query = state["context"]["query"]

    response = llm.llm_normal.invoke(
        [
            SystemMessage(
                content=f"""You classify a user message based on query as either STORE or FETCH.

STORE = the user is giving you information about a stock purchase to save (any details, any format).
FETCH = the user is asking to see/retrieve previously saved holdings info.

Reply with exactly one word: STORE or FETCH. here is the query : {query}""",
            ),
            *state["messages"],
        ]
    )

    intent = response.content.strip().upper()
    if "FETCH" in intent:
        intent = "FETCH"
    else:
        intent = "STORE"

    return {"intent": intent}


def route(state: PortfolioManagerAgent):
    return state["intent"].lower()


# ---- Node 2a: store raw text ----
def store_node(state: PortfolioManagerAgent):
    content = state["context"]["query"]

    conn = get_connection()
    conn.execute(
        "INSERT INTO holdings_text (content, created_at) VALUES (?, ?)",
        (content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

    reply = "Saved."
    return {"messages": [{"role": "assistant", "content": reply}], "response": reply}


# ---- Node 2b: fetch raw text ----
def fetch_node(state: PortfolioManagerAgent):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content, created_at FROM holdings_text ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    query = state["context"]["query"]
    if not rows:
        reply = "No holdings saved yet."
    else:
        reply = "\n\n".join(f"[{ts}]\n{content}" for content, ts in rows)

    response = llm.llm_normal.invoke(f"""answer to this question :{query} 
    based on provided info fetched from db: {reply}""").content

    return {"messages": [{"role": "assistant", "content": response}], "response": response}


# ---- Graph ----
builder = StateGraph(PortfolioManagerAgent)

builder.add_node("classify", classify_intent)
builder.add_node("store", store_node)
builder.add_node("fetch", fetch_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {"store": "store", "fetch": "fetch"},
)
builder.add_edge("store", END)
builder.add_edge("fetch", END)

graph = builder.compile(checkpointer=InMemorySaver())