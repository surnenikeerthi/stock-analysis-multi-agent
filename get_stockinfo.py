





#ignore




from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
import yfinance as yf
import llm
class stock_info(TypedDict):
    stock_name : str
    ticker_obj : Any
    stock_financial: Any
    balance_sheet: Any
    stock_history: Any

def stock_get_ticker_obj(state: stock_info):
    stock = state["stock"]
    ticker_obj = yf.Ticker(stock)
    stock_financial = get_stock_financials(ticker_obj)
    balance_sheet = get_stock_balance_sheet(ticker_obj)
    stock_history = get_stock_history(ticker_obj)
    '''print(type(stock_financial))
    print(type(balance_sheet))
    print(ticker_obj)'''
    return {'ticker_obj':ticker_obj, 'stock_financial' : stock_financial, 'balance_sheet':balance_sheet}

def get_stock_financials(stock_ticker_obj):
    stock_financial = stock_ticker_obj.get_financials()
    #print(stock_financial)
    return stock_financial

def get_stock_balance_sheet(stock_ticker_obj):
    balance_sheet = stock_ticker_obj.get_balance_sheet()
    return balance_sheet

def get_stock_history(stock_ticker_obj):
    stock_history = stock_ticker_obj.get_history()
    return stock_history

def agent_decider():
    prompt = "which agen shall be caled income statement or what "


def stock_analysis(state: prompt_orchestrator):
    prompt = ("You are responsible for routing a stock analysis query."
              "User Query:"
              "{query}"

              " Available agents:"
              "- income_statement_analysis_agent"
              "- balance_sheet_analysis_agent"
              "- cash_flow_analysis_agent"
              "- valuation_analysis_agent"
              "- technical_analysis_agent"
              "- news_sentiment_analysis_agent"

              "Return ONLY valid JSON in the following format:"

              " {"
              "agents: ["
              "income_statement_analysis_agent",
              "valuation_analysis_agent"
              " ]"
              " }"

              "The agents array must contain only the agents required to answer the query and must be in execution order.")
    agents = llm_normal.invoke(prompt).content
    for agent in agents:
        agent.graph.invoke(query)























def data_fetcher(finance_statement, parameter):
    print(parameter)
    revenue = finance_statement.loc[parameter]
    data = [revenue.iloc[i] for i in range(3)]
    return data

# node that gets the revenue details
def revenue_fetcher(state:stock_info):
    revenue_list_for_3_years = data_fetcher(state['stock_financial'], 'TotalRevenue')
    return {'revenue_list_for_3_years' :revenue_list_for_3_years}

#used both for net income and profit
def get_net_profit(state:stock_info):
    profit_list_for_3_years = data_fetcher(state['stock_financial'], 'NetIncome')
    return {'profit_list_for_3_years' :profit_list_for_3_years}

def get_EPS(state:stock_info):
    EPS_list_for_3_years = data_fetcher(state['stock_financial'], 'BasicEPS ')
    return {'EPS_list_for_3_years' : EPS_list_for_3_years}

# get the PE ratio (price to earning ratio)
def get_PE_ratio(state:stock_info):
    PE_ratio = stock_info(state['ticker_obj'], 'forwardPE')
    return {'PE_ratio' :PE_ratio}

def return_of_equity(state:stock_info):
    ROE = stock_info(state['ticker_obj'], 'returnOnEquity')
    return {'ROE' :ROE}

def debt_to_equity(state:stock_info):
    balance_sheet = state['balance_sheet']
    total_debt = data_fetcher(balance_sheet, "TotalDebt")
    equity = data_fetcher(balance_sheet,"StockholdersEquity")
    de_ratio = [debt/eq for debt,eq in zip(total_debt,equity)]
    return {'de_ratio':de_ratio, 'total_debt':total_debt, 'equity':equity}

def free_cash_flow(state:stock_info):
    stock = state['ticker_obj']
    FCF = stock_info(stock, 'Free Cash Flow')
    if (FCF == 'NAN'):
        operating_cashflow =  stock_info(stock, 'OperatingCashFlow')
        capital_Expenditure = stock.info(stock, 'CapitalExpenditure')
        capital_Expenditure = (capital_Expenditure*-1) if(capital_Expenditure < 0) else capital_Expenditure
        FCF = operating_cashflow - capital_Expenditure
    return {'FCF' :FCF}

def get_gross_profit(state: stock_info):
    gross_profit = data_fetcher(state['stock_financial'], 'GrossProfit')
    return {'gross_profit': gross_profit}

def get_operating_income(state: stock_info):
    operating_income = data_fetcher(state['stock_financial'], 'OperatingIncome')
    return {'operating_income': operating_income}

def get_ebitda(state: stock_info):
    EBITDA = data_fetcher(state['stock_financial'], 'EBITDA')
    return {'EBITDA': EBITDA}

def gross_margin(state: stock_info):

    revenue = data_fetcher(state['stock_financial'], 'TotalRevenue')
    gross_profit = data_fetcher(state['stock_financial'], 'GrossProfit')

    margin = [(gp/rev)*100 for gp, rev in zip(gross_profit, revenue)]

    return {'gross_margin': margin}

def operating_margin(state: stock_info):

    revenue = data_fetcher(state['stock_financial'], 'TotalRevenue')
    operating_income = data_fetcher(state['stock_financial'], 'OperatingIncome')

    margin = [(op/rev)*100 for op, rev in zip(operating_income, revenue)]

    return {'operating_margin': margin}

def net_profit_margin(state: stock_info):

    revenue = data_fetcher(state['stock_financial'], 'TotalRevenue')
    net_income = data_fetcher(state['stock_financial'], 'NetIncome')

    margin = [(profit/rev)*100 for profit, rev in zip(net_income, revenue)]

    return {'net_profit_margin': margin}

def revenue_growth(state: stock_info):

    revenue = data_fetcher(state['stock_financial'], 'TotalRevenue')

    growth = []

    for i in range(len(revenue)-1):
        growth.append(((revenue[i]-revenue[i+1])/revenue[i+1])*100)

    return {'revenue_growth': growth}

def net_income_growth(state: stock_info):

    net_income = data_fetcher(state['stock_financial'], 'NetIncome')

    growth = []

    for i in range(len(net_income)-1):
        growth.append(((net_income[i]-net_income[i+1])/net_income[i+1])*100)

    return {'net_income_growth': growth}

def EPS_growth(state: stock_info):

    EPS = data_fetcher(state['stock_financial'], 'BasicEPS')

    growth = []

    for i in range(len(EPS)-1):
        growth.append(((EPS[i]-EPS[i+1])/EPS[i+1])*100)

    return {'EPS_growth': growth}

def return_on_capital_employed(state:stock_info):
    balance_sheet = state['balance_sheet']
    stock_financial = state['stock_financial']
    ebit = data_fetcher(stock_financial, "OperatingIncome")
    if(ebit[0] == 'NAN'):
        return {'ROCE' :''}
    print("ebit")
    print(ebit)
    current_liabilities = data_fetcher(balance_sheet, "CurrentLiabilities")
    total_assets = data_fetcher(balance_sheet, "TotalAssets")

    capital_employed = [(asset - liability) for asset, liability in zip(total_assets,current_liabilities)]
    ROCE = [(ebit / capital_employed)for ebit, capital in zip(ebit,capital_employed)]
    return {'ROCE' :ROCE}

info_builder = StateGraph(stock_info)
info_builder.add_node("stock_get_ticker_obj", stock_get_ticker_obj)
info_builder.add_edge(START, 'stock_get_ticker_obj')
graph = info_builder.compile()