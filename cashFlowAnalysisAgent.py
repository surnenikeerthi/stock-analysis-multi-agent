import stock_analyser
from typing import Any, TypedDict
import llm
from langgraph.graph import StateGraph, START, END


class cashFlowAnalysisAgent(TypedDict):
    context: Any
    operating_cashflow: Any
    capital_expenditure: Any
    FCF: Any
    FCF_growth: Any
    operating_cashflow_growth: Any
    capital_expenditure_trend: Any
    FCF_margin: Any
    analysis: Any  # final LLM output


# ---------------------------------------------------------------------
# safety helpers - data_fetcher returns None (not a raised exception)
# when a row label is missing, and lists may be shorter than expected -
# every computed metric below needs to tolerate that.
# ---------------------------------------------------------------------

def _compute_fcf(ticker):
    """Fetch Free Cash Flow directly if yfinance reports it; otherwise
    derive it as OperatingCashFlow - |CapitalExpenditure|. Returns None
    if neither path has usable data."""
    FCF = stock_analyser.data_fetcher(ticker, 'cash_flow', 'FreeCashFlow')
    if FCF is not None:
        return FCF

    operating_cashflow = stock_analyser.data_fetcher(ticker, 'cash_flow', 'OperatingCashFlow')
    capex = stock_analyser.data_fetcher(ticker, 'cash_flow', 'CapitalExpenditure')

    if operating_cashflow is None or capex is None:
        return None

    return [
        (ocf - abs(c)) if (ocf is not None and c is not None) else None
        for ocf, c in zip(operating_cashflow, capex)
    ]


def _safe_pct_change_list(series):
    """Period-over-period % change: [(x0-x1)/x1, (x1-x2)/x2, ...] * 100.
    Tolerant of None entries, zero denominators, and None/empty series."""
    if not series:
        return None
    out = []
    for i in range(len(series) - 1):
        curr, nxt = series[i], series[i + 1]
        if curr is None or nxt in (None, 0):
            out.append(None)
        else:
            out.append(((curr - nxt) / nxt) * 100)
    return out


def _safe_ratio_list(numerators, denominators, multiplier=1):
    """Elementwise numerators/denominators * multiplier, tolerant of
    None inputs/values, zero denominators, and mismatched lengths."""
    if not numerators or not denominators:
        return None
    return [
        (n / d) * multiplier if (n is not None and d not in (None, 0)) else None
        for n, d in zip(numerators, denominators)
    ]


# Operating Cash Flow
def operating_cash_flow(state: cashFlowAnalysisAgent):
    operating_cashflow = stock_analyser.data_fetcher(state['context']['ticker'], 'cash_flow', 'OperatingCashFlow')
    return {'operating_cashflow': operating_cashflow}


# Capital Expenditure (CapEx)
def capital_expenditure(state: cashFlowAnalysisAgent):
    capex = stock_analyser.data_fetcher(state['context']['ticker'], 'cash_flow', 'CapitalExpenditure')
    return {'capital_expenditure': capex}


# Free Cash Flow
def free_cash_flow(state: cashFlowAnalysisAgent):
    FCF = _compute_fcf(state['context']['ticker'])
    return {'FCF': FCF}


# Free Cash Flow Growth (reuses FCF already computed by the free_cash_flow node)
def free_cash_flow_growth(state: cashFlowAnalysisAgent):
    growth = _safe_pct_change_list(state.get('FCF'))
    return {'FCF_growth': growth}


# Operating Cash Flow Growth
def operating_cash_flow_growth(state: cashFlowAnalysisAgent):
    operating_cashflow = stock_analyser.data_fetcher(state['context']['ticker'], 'cash_flow', 'OperatingCashFlow')
    growth = _safe_pct_change_list(operating_cashflow)
    return {'operating_cashflow_growth': growth}


# Capital Expenditure Trend (period-over-period % change, not a raw duplicate of CapEx)
def capital_expenditure_trend(state: cashFlowAnalysisAgent):
    capex = stock_analyser.data_fetcher(state['context']['ticker'], 'cash_flow', 'CapitalExpenditure')
    trend = _safe_pct_change_list(capex)
    return {'capital_expenditure_trend': trend}


# Free Cash Flow Margin (reuses FCF already computed by the free_cash_flow node)
def free_cash_flow_margin(state: cashFlowAnalysisAgent):
    ticker = state['context']['ticker']
    revenue = stock_analyser.data_fetcher(ticker, 'stock_financial', 'TotalRevenue')
    margin = _safe_ratio_list(state.get('FCF'), revenue, multiplier=100)
    return {'FCF_margin': margin}


# Final synthesis node
def analyse_cash_flow(state: cashFlowAnalysisAgent):
    prompt = f"""You are an expert financial analyst specializing in cash flow analysis.

        Analyze the following cash flow data for {state['context']['ticker']} over the last three reporting periods (most recent first).

        Cash Flow Data:
        - Operating Cash Flow: {state.get('operating_cashflow')}
        - Capital Expenditure: {state.get('capital_expenditure')}
        - Free Cash Flow: {state.get('FCF')}
        - Free Cash Flow Growth (%): {state.get('FCF_growth')}
        - Operating Cash Flow Growth (%): {state.get('operating_cashflow_growth')}
        - Capital Expenditure Trend (%): {state.get('capital_expenditure_trend')}
        - Free Cash Flow Margin (%): {state.get('FCF_margin')}

        Some fields may be null/None where the underlying data is not disclosed or could
        not be computed - treat those as "not available" rather than zero, and do not
        fabricate a value.

        Evaluate the company's:
        - Operating cash flow trend and consistency
        - Free cash flow generation and sustainability
        - Capital expenditure trend and investment strategy
        - Cash conversion efficiency
        - Overall cash flow quality and financial flexibility

        Return ONLY valid JSON in the following format:

        {{
          "agent": "cashflow",

          "metrics": {{
            // Include only the most relevant metrics and values that support your conclusions.
          }},

          "analysis": "Provide a concise professional analysis explaining your reasoning based on the cash flow data.",

          "conclusion": {{
            "sentiment": "Bullish | Neutral | Bearish",
            "summary": "A one-sentence summary of the company's cash flow health."
          }}
        }}

        Rules:
        - Return ONLY valid JSON.
        - Do not include markdown or additional text.
        - Base every conclusion strictly on the supplied data.
        - Include only metrics that materially support your analysis.
        - Highlight both positive and negative observations when applicable.
        - Do not invent values or assumptions.
        - Omit metrics that cannot be determined.
        - Keep the analysis concise (3–6 sentences)."""
    response = llm.llm_normal.invoke(prompt)
    return {'analysis': response.content}


builder = StateGraph(cashFlowAnalysisAgent)

builder.add_node("operating_cash_flow", operating_cash_flow)
builder.add_node("capital_expenditure", capital_expenditure)
builder.add_node("free_cash_flow", free_cash_flow)
builder.add_node("free_cash_flow_growth", free_cash_flow_growth)
builder.add_node("operating_cash_flow_growth", operating_cash_flow_growth)
builder.add_node("capital_expenditure_trend", capital_expenditure_trend)
builder.add_node("free_cash_flow_margin", free_cash_flow_margin)
builder.add_node("analyse_cash_flow", analyse_cash_flow)

# Fan-out: nodes that only need context.ticker run in parallel off START
for node in [
    "operating_cash_flow", "capital_expenditure", "free_cash_flow",
    "operating_cash_flow_growth", "capital_expenditure_trend",
]:
    builder.add_edge(START, node)

# free_cash_flow_growth and free_cash_flow_margin reuse the FCF already
# computed by free_cash_flow instead of re-deriving it independently
builder.add_edge("free_cash_flow", "free_cash_flow_growth")
builder.add_edge("free_cash_flow", "free_cash_flow_margin")

# Fan-in: final analysis waits for all of them
for node in [
    "operating_cash_flow", "capital_expenditure", "free_cash_flow",
    "free_cash_flow_growth", "operating_cash_flow_growth",
    "capital_expenditure_trend", "free_cash_flow_margin",
]:
    builder.add_edge(node, "analyse_cash_flow")

builder.add_edge("analyse_cash_flow", END)

graph = builder.compile()