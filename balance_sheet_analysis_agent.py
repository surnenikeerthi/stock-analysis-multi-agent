import stock_analyser
from typing import Any, TypedDict
import llm
from langgraph.graph import StateGraph, START, END


class balance_sheet_analysis_agent(TypedDict):
    context: Any
    total_assets: Any
    total_liabilities: Any
    total_equity: Any
    current_assets: Any
    current_liabilities: Any
    cash_and_cash_equivalents: Any
    total_debt: Any
    long_term_debt: Any
    current_ratio: Any
    quick_ratio: Any
    ROA: Any

    analysis: Any  # final LLM output


# ---------------------------------------------------------------------
# safety helpers - data_fetcher can now return None (missing row label)
# or lists of unequal length, so every computed ratio needs to tolerate
# that instead of crashing on zip()/division.
# ---------------------------------------------------------------------

def _safe_div_list(numerators, denominators):
    """Elementwise division that tolerates None inputs/values and
    mismatched lengths instead of raising."""
    if not numerators or not denominators:
        return None
    return [
        (n / d) if (n is not None and d not in (None, 0)) else None
        for n, d in zip(numerators, denominators)
    ]


def analyse_balance_sheet(state: balance_sheet_analysis_agent):
    prompt = f"""You are an expert financial analyst specializing in balance sheet analysis.

        Analyze the following balance sheet data for {state['context']['ticker']} over the last three reporting periods (most recent first).

        Balance Sheet Data:
        - Total Assets: {state.get('total_assets')}
        - Total Liabilities: {state.get('total_liabilities')}
        - Shareholders' Equity: {state.get('total_equity')}
        - Current Assets: {state.get('current_assets')}
        - Current Liabilities: {state.get('current_liabilities')}
        - Cash & Cash Equivalents: {state.get('cash_and_cash_equivalents')}
        - Total Debt: {state.get('total_debt')}
        - Long-Term Debt: {state.get('long_term_debt')}
        - Current Ratio: {state.get('current_ratio')}
        - Quick Ratio: {state.get('quick_ratio')}
        - Return on Assets (ROA): {state.get('ROA')}

        Some fields may be null/None where the company does not disclose that line item -
        treat those as "not disclosed" rather than zero, and do not fabricate a value.

        Evaluate the company's:
        - Asset growth and quality
        - Liability and debt trends
        - Equity strength and solvency
        - Liquidity position
        - Overall balance sheet health

        Return ONLY valid JSON in the following format:

        {{
          "agent": "balance_sheet",

          "metrics": {{
            // Include only the most relevant metrics from above and values that support your conclusions.
          }},

          "analysis": "Provide a concise professional analysis explaining your reasoning based on the balance sheet data provided in less than 100 words.",

          "conclusion": {{
            "sentiment": "Bullish | Neutral | Bearish",
            "summary": "A one-sentence summary of the balance sheet health."
          }}
        }}

        Rules:
        - Return ONLY valid JSON.
        - Do not include markdown or additional text.
        - Base every conclusion strictly on the supplied data.
        - Include only metrics that materially support your analysis.
        - Do not invent values or assumptions.
        - Omit metrics that cannot be determined.
        - Keep the analysis concise (3-6 sentences)."""
    response = llm.llm_normal.invoke(prompt)
    return {'analysis': response.content}


# Total Assets
def get_total_assets(state: balance_sheet_analysis_agent):
    total_assets = stock_analyser.data_fetcher(state['context']['ticker'], 'balance_sheet', 'TotalAssets')
    return {'total_assets': total_assets}


# Total Liabilities
def get_total_liabilities(state: balance_sheet_analysis_agent):
    total_liabilities = stock_analyser.data_fetcher(
        state['context']['ticker'], 'balance_sheet', 'TotalLiabilitiesNetMinorityInterest'
    )
    return {'total_liabilities': total_liabilities}


# Shareholders' Equity
def get_total_equity(state: balance_sheet_analysis_agent):
    total_equity = stock_analyser.data_fetcher(state['context']['ticker'], 'balance_sheet', 'StockholdersEquity')
    return {'total_equity': total_equity}


# Current Assets
def get_current_assets(state: balance_sheet_analysis_agent):
    current_assets = stock_analyser.data_fetcher(state['context']['ticker'], 'balance_sheet', 'CurrentAssets')
    return {'current_assets': current_assets}


# Current Liabilities
def get_current_liabilities(state: balance_sheet_analysis_agent):
    current_liabilities = stock_analyser.data_fetcher(
        state['context']['ticker'], 'balance_sheet', 'CurrentLiabilities'
    )
    return {'current_liabilities': current_liabilities}


# Cash and Cash Equivalents
def get_cash_and_cash_equivalents(state: balance_sheet_analysis_agent):
    cash = stock_analyser.data_fetcher(
        state['context']['ticker'], 'balance_sheet', 'CashAndCashEquivalents'
    )
    return {'cash_and_cash_equivalents': cash}


# Total Debt
def get_total_debt(state: balance_sheet_analysis_agent):
    total_debt = stock_analyser.data_fetcher(state['context']['ticker'], 'balance_sheet', 'TotalDebt')
    return {'total_debt': total_debt}


# Long Term Debt
def get_long_term_debt(state: balance_sheet_analysis_agent):
    long_term_debt = stock_analyser.data_fetcher(
        state['context']['ticker'], 'balance_sheet', 'LongTermDebt'
    )
    if long_term_debt is None:
        # yfinance uses inconsistent labels across companies - try known synonyms
        long_term_debt = stock_analyser.data_fetcher(
            state['context']['ticker'], 'balance_sheet', 'LongTermDebtNoncurrent'
        )
    if long_term_debt is None:
        long_term_debt = stock_analyser.data_fetcher(
            state['context']['ticker'], 'balance_sheet', 'LongTermDebtAndCapitalLeaseObligation'
        )
    return {'long_term_debt': long_term_debt}


# Current Ratio
def current_ratio(state: balance_sheet_analysis_agent):
    ratio = _safe_div_list(state.get('current_assets'), state.get('current_liabilities'))
    return {'current_ratio': ratio}


# Quick Ratio
def quick_ratio(state: balance_sheet_analysis_agent):
    cash = state.get('cash_and_cash_equivalents')
    current_assets = state.get('current_assets')
    current_liabilities = state.get('current_liabilities')
    inventory = stock_analyser.data_fetcher(state['context']['ticker'], 'balance_sheet', 'Inventory')

    if not current_assets or not current_liabilities or not cash:
        return {'quick_ratio': None}

    # inventory may be None (company doesn't hold/report inventory, e.g. many
    # service/financial companies) - treat missing inventory as 0 rather
    # than failing the whole calculation.
    if inventory is None:
        inventory = [0] * len(current_assets)

    quick_assets = [
        (asset - inv + c) if None not in (asset, inv, c) else None
        for asset, inv, c in zip(current_assets, inventory, cash)
    ]

    ratio = _safe_div_list(quick_assets, current_liabilities)
    return {'quick_ratio': ratio}


# Return on Assets (ROA)
def return_on_assets(state: balance_sheet_analysis_agent):
    net_income = stock_analyser.data_fetcher(state['context']['ticker'], 'stock_financial', 'NetIncome')
    total_assets = state.get('total_assets')

    if not net_income or not total_assets:
        return {'ROA': None}

    roa = [
        (income / assets) * 100 if (income is not None and assets not in (None, 0)) else None
        for income, assets in zip(net_income, total_assets)
    ]
    return {'ROA': roa}


builder = StateGraph(balance_sheet_analysis_agent)

# Raw data fetch nodes — all independent, run in parallel
builder.add_node("get_total_assets", get_total_assets)
builder.add_node("get_total_liabilities", get_total_liabilities)
builder.add_node("get_total_equity", get_total_equity)
builder.add_node("get_current_assets", get_current_assets)
builder.add_node("get_current_liabilities", get_current_liabilities)
builder.add_node("get_cash_and_cash_equivalents", get_cash_and_cash_equivalents)
builder.add_node("get_total_debt", get_total_debt)
builder.add_node("get_long_term_debt", get_long_term_debt)

# Derived/computed nodes — depend on specific fetch nodes above
builder.add_node("current_ratio", current_ratio)
builder.add_node("quick_ratio", quick_ratio)
builder.add_node("return_on_assets", return_on_assets)

# Final synthesis node
builder.add_node("analyse_balance_sheet", analyse_balance_sheet)

# Fan-out: START -> all raw fetchers (parallel)
for fetch_node in [
    "get_total_assets", "get_total_liabilities", "get_total_equity",
    "get_current_assets", "get_current_liabilities",
    "get_cash_and_cash_equivalents", "get_total_debt", "get_long_term_debt",
]:
    builder.add_edge(START, fetch_node)

# current_ratio needs current_assets + current_liabilities
builder.add_edge("get_current_assets", "current_ratio")
builder.add_edge("get_current_liabilities", "current_ratio")

# quick_ratio needs current_assets + current_liabilities + cash (inventory fetched inline)
builder.add_edge("get_current_assets", "quick_ratio")
builder.add_edge("get_current_liabilities", "quick_ratio")
builder.add_edge("get_cash_and_cash_equivalents", "quick_ratio")

# return_on_assets needs total_assets
builder.add_edge("get_total_assets", "return_on_assets")

# Everything feeds into the final analysis node
# (get_total_assets is included explicitly even though it's already an
# ancestor of return_on_assets - the analysis node reads state['total_assets']
# directly in its prompt, so it should not rely on transitive ordering
# through an unrelated node to guarantee that value is populated first.)
for node in [
    "get_total_assets", "get_total_liabilities", "get_total_equity",
    "get_total_debt", "get_long_term_debt",
    "current_ratio", "quick_ratio", "return_on_assets",
]:
    builder.add_edge(node, "analyse_balance_sheet")

builder.add_edge("analyse_balance_sheet", END)

graph = builder.compile()