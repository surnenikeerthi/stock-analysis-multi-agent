import stock_analyser
from typing import Any, TypedDict
import llm
from langgraph.graph import StateGraph, START, END

class income_statement_agent(TypedDict):
    context: Any
    revenue_list_for_3_years: Any
    profit_list_for_3_years: Any
    EPS_list_for_3_years: Any
    operating_income: Any
    EBITDA: Any
    gross_margin: Any
    operating_margin: Any
    net_profit_margin: Any
    revenue_growth: Any
    net_income_growth: Any
    EPS_growth: Any
    analysis: Any  # final LLM output

def revenue_fetcher(state: income_statement_agent):
    revenue_list_for_3_years = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'TotalRevenue')
    return {'revenue_list_for_3_years': revenue_list_for_3_years}


def get_net_profit(state: income_statement_agent):
    profit_list_for_3_years = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'NetIncome')
    return {'profit_list_for_3_years': profit_list_for_3_years}


def get_EPS(state: income_statement_agent):
    EPS_list_for_3_years = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'BasicEPS')
    return {'EPS_list_for_3_years': EPS_list_for_3_years}


def get_operating_income(state: income_statement_agent):
    operating_income = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'OperatingIncome')
    return {'operating_income': operating_income}


def get_ebitda(state: income_statement_agent):
    EBITDA = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'EBITDA')
    return {'EBITDA': EBITDA}


def gross_margin(state: income_statement_agent):
    ticker = state['context']['ticker']
    revenue = stock_analyser.data_fetcher(ticker, 'stock_financial', 'TotalRevenue')
    gross_profit = stock_analyser.data_fetcher(ticker, 'stock_financial', 'GrossProfit')
    margin = [(gp / rev) * 100 for gp, rev in zip(gross_profit, revenue)]
    return {'gross_margin': margin}


def operating_margin(state: income_statement_agent):
    ticker = state['context']['ticker']
    revenue = stock_analyser.data_fetcher(ticker, 'stock_financial', 'TotalRevenue')
    operating_income = stock_analyser.data_fetcher(ticker, 'stock_financial', 'OperatingIncome')
    margin = [(op / rev) * 100 for op, rev in zip(operating_income, revenue)]
    return {'operating_margin': margin}


def net_profit_margin(state: income_statement_agent):
    ticker = state['context']['ticker']
    revenue = stock_analyser.data_fetcher(ticker, 'stock_financial', 'TotalRevenue')
    net_income = stock_analyser.data_fetcher(ticker, 'stock_financial', 'NetIncome')
    margin = [(profit / rev) * 100 for profit, rev in zip(net_income, revenue)]
    return {'net_profit_margin': margin}


def revenue_growth(state: income_statement_agent):
    revenue = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'TotalRevenue')
    growth = [((revenue[i] - revenue[i + 1]) / revenue[i + 1]) * 100 for i in range(len(revenue) - 1)]
    return {'revenue_growth': growth}


def net_income_growth(state: income_statement_agent):
    net_income = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'NetIncome')
    growth = [((net_income[i] - net_income[i + 1]) / net_income[i + 1]) * 100 for i in range(len(net_income) - 1)]
    return {'net_income_growth': growth}


def EPS_growth(state: income_statement_agent):
    EPS = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'BasicEPS')
    growth = [((EPS[i] - EPS[i + 1]) / EPS[i + 1]) * 100 for i in range(len(EPS) - 1)]
    return {'EPS_growth': growth}


def analyse_income_statement(state: income_statement_agent):
    prompt = f"""
You are a financial analyst. Analyze the following income statement data for {state['context']['ticker']}
across the last 3 reporting periods (most recent first):

Revenue: {state['revenue_list_for_3_years']}
Net Income: {state['profit_list_for_3_years']}
Basic EPS: {state['EPS_list_for_3_years']}
Operating Income: {state['operating_income']}
EBITDA: {state['EBITDA']}
Gross Margin (%): {state['gross_margin']}
Operating Margin (%): {state['operating_margin']}
Net Profit Margin (%): {state['net_profit_margin']}
Revenue Growth (%): {state['revenue_growth']}
Net Income Growth (%): {state['net_income_growth']}
EPS Growth (%): {state['EPS_growth']}

Based on this data:
1. Assess the trend in revenue and net income over the 3 periods.
2. Evaluate margin trends (gross, operating, net) — improving, stable, or deteriorating?
3. Assess earnings quality using EPS growth alongside net income growth.
4. Give an overall verdict on income statement health (Strong / Moderate / Weak), with reasoning.

Respond in a structured format with clear headers for each section.
"""
    response = llm.llm_normal.invoke(prompt)
    return {'analysis': response.content}

builder = StateGraph(income_statement_agent)

metric_nodes = {
    "revenue_fetcher": revenue_fetcher,
    "get_net_profit": get_net_profit,
    "get_EPS": get_EPS,
    "get_operating_income": get_operating_income,
    "get_ebitda": get_ebitda,
    "gross_margin": gross_margin,
    "operating_margin": operating_margin,
    "net_profit_margin": net_profit_margin,
    "revenue_growth": revenue_growth,
    "net_income_growth": net_income_growth,
    "EPS_growth": EPS_growth,
}

for name, fn in metric_nodes.items():
    builder.add_node(name, fn)

builder.add_node("analyse_income_statement", analyse_income_statement)

for name in metric_nodes:
    builder.add_edge(START, name)
    builder.add_edge(name, "analyse_income_statement")

builder.add_edge("analyse_income_statement", END)

graph = builder.compile()