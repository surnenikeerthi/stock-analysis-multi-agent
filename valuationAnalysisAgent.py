import yfinance as yf
from typing import Any, TypedDict
import llm
from langgraph.graph import StateGraph, START, END

class valuationAnalysisAgent(TypedDict):
    context: Any
    market_cap: Any
    PE_ratio: Any
    trailing_PE: Any
    PEG_ratio: Any
    price_to_book: Any
    price_to_sales: Any
    enterprise_value: Any
    EV_to_EBITDA: Any
    EV_to_revenue: Any
    dividend_yield: Any
    book_value_per_share: Any
    earnings_yield: Any
    analysis: Any  # final LLM output

def _get_info(state: valuationAnalysisAgent):
    """Shared helper — not a node itself, just avoids repeating yf.Ticker(...) in every node."""
    return yf.Ticker(state['context']['ticker']).info


def market_cap(state: valuationAnalysisAgent):
    return {'market_cap': _get_info(state).get("marketCap")}


def get_PE_ratio(state: valuationAnalysisAgent):
    return {'PE_ratio': _get_info(state).get("forwardPE")}


def trailing_PE(state: valuationAnalysisAgent):
    return {'trailing_PE': _get_info(state).get("trailingPE")}


def PEG_ratio(state: valuationAnalysisAgent):
    return {'PEG_ratio': _get_info(state).get("pegRatio")}


def price_to_book(state: valuationAnalysisAgent):
    return {'price_to_book': _get_info(state).get("priceToBook")}


def price_to_sales(state: valuationAnalysisAgent):
    return {'price_to_sales': _get_info(state).get("priceToSalesTrailing12Months")}


def enterprise_value(state: valuationAnalysisAgent):
    return {'enterprise_value': _get_info(state).get("enterpriseValue")}


def EV_to_EBITDA(state: valuationAnalysisAgent):
    return {'EV_to_EBITDA': _get_info(state).get("enterpriseToEbitda")}


def EV_to_revenue(state: valuationAnalysisAgent):
    return {'EV_to_revenue': _get_info(state).get("enterpriseToRevenue")}


def dividend_yield(state: valuationAnalysisAgent):
    return {'dividend_yield': _get_info(state).get("dividendYield")}


def book_value_per_share(state: valuationAnalysisAgent):
    return {'book_value_per_share': _get_info(state).get("bookValue")}


def earnings_yield(state: valuationAnalysisAgent):
    pe = _get_info(state).get("trailingPE")
    if pe is None or pe == 0:
        return {'earnings_yield': None}
    return {'earnings_yield': (1 / pe) * 100}


def analyse_valuation(state: valuationAnalysisAgent):
    prompt = f"""
You are a financial analyst. Analyze the following current valuation metrics for {state['context']['ticker']}:

Market Cap: {state['market_cap']}
Forward PE Ratio: {state['PE_ratio']}
Trailing PE Ratio: {state['trailing_PE']}
PEG Ratio: {state['PEG_ratio']}
Price to Book: {state['price_to_book']}
Price to Sales: {state['price_to_sales']}
Enterprise Value: {state['enterprise_value']}
EV/EBITDA: {state['EV_to_EBITDA']}
EV/Revenue: {state['EV_to_revenue']}
Dividend Yield: {state['dividend_yield']}
Book Value Per Share: {state['book_value_per_share']}
Earnings Yield: {state['earnings_yield']}

Based on this data:
1. Assess whether the stock appears overvalued, fairly valued, or undervalued based on PE, PEG, and EV/EBITDA.
2. Comment on what Price-to-Book and Price-to-Sales suggest about how the market values the company's assets and revenue.
3. Evaluate dividend yield and earnings yield in the context of income-focused investing.
4. Give an overall verdict on valuation attractiveness (Attractive / Fair / Expensive), with reasoning.
Note: these are single point-in-time snapshot metrics, not multi-period trends — frame the analysis accordingly.

Respond in a structured format with clear headers for each section.
"""
    response = llm.llm_normal.invoke(prompt)
    return {'analysis': response.content}

builder = StateGraph(valuationAnalysisAgent)

metric_nodes = {
    "market_cap": market_cap,
    "get_PE_ratio": get_PE_ratio,
    "trailing_PE": trailing_PE,
    "PEG_ratio": PEG_ratio,
    "price_to_book": price_to_book,
    "price_to_sales": price_to_sales,
    "enterprise_value": enterprise_value,
    "EV_to_EBITDA": EV_to_EBITDA,
    "EV_to_revenue": EV_to_revenue,
    "dividend_yield": dividend_yield,
    "book_value_per_share": book_value_per_share,
    "earnings_yield": earnings_yield,
}

for name, fn in metric_nodes.items():
    builder.add_node(name, fn)

builder.add_node("analyse_valuation", analyse_valuation)

# Fan-out: all metrics only need context.ticker
for name in metric_nodes:
    builder.add_edge(START, name)

# Fan-in: final analysis waits for all 12
for name in metric_nodes:
    builder.add_edge(name, "analyse_valuation")

builder.add_edge("analyse_valuation", END)

graph = builder.compile()