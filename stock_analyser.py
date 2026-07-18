from typing import TypedDict, Any
import yfinance as yf
import llm
import numpy
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
import llm_chains
import balance_sheet_analysis_agent,income_statement_agent,technical_agent,newsAnalysisAgent,cashFlowAnalysisAgent,PortfolioManagerAgent
from llm import llm_normal

agent_registry = {
    "income_statement_agent": income_statement_agent,
    "balance_sheet_analysis_agent": balance_sheet_analysis_agent,
    "cashFlowAnalysisAgent": cashFlowAnalysisAgent,
    "technical_agent": technical_agent,
    "newsAnalysisAgent": newsAnalysisAgent,
    "PortfolioManagerAgent": PortfolioManagerAgent
}

class stock_analyser(TypedDict):
    context :Any
    full_information : str
    response :str

def data_fetcher(stock_ticker, container, parameter):

    stock_ticker_obj = yf.Ticker(stock_ticker)
    statement = ""
    if (container == 'stock_financial'):
        statement = get_stock_financials(stock_ticker_obj)
    if (container == 'balance_sheet'):
        statement = get_stock_balance_sheet(stock_ticker_obj)
    if (container == 'cash_flow'):
        statement = get_stock_cash_flow(stock_ticker_obj)
    if (container == 'price_history'):
        statement = stock_ticker_obj.history(period='1mo', interval='1d')

    if isinstance(statement, str):
        raise ValueError(
            f"data_fetcher: unknown container '{container}'. "
            f"Expected one of: 'stock_financial', 'balance_sheet', 'cash_flow', 'price_history'."
        )

    #print(statement)

    try:
        revenue = statement.loc[parameter]
    except KeyError:
        print(f"[data_fetcher] '{parameter}' not found in {container} for {stock_ticker}. "
              f"Available rows: {list(statement.index)}")
        return None

    try:
        if isinstance(revenue.iloc[0], numpy.float64):
            data = [float(revenue.iloc[i]) for i in range(3)]
        else:
            data = [revenue.iloc[i] for i in range(3)]
    except IndexError:
        print(f"[data_fetcher] '{parameter}' for {stock_ticker} has fewer than 3 periods of data "
              f"(only {len(revenue)} available).")
        data = [float(v) if isinstance(v, numpy.float64) else v for v in revenue.tolist()]

    return data


def get_stock_cash_flow(stock_ticker_obj):
    return stock_ticker_obj.cash_flow

#node to get the finance and make decision on is it in good case or not
def get_stock_financials(stock_ticker_obj):
    stock_financial = stock_ticker_obj.get_financials()
    #print(stock_financial)
    return stock_financial

def get_stock_balance_sheet(stock_ticker_obj):
    balance_sheet = stock_ticker_obj.get_balance_sheet()
    return balance_sheet

def get_stock_history(stock_ticker_obj):
    history = stock_ticker_obj.get_history()
    return history

def stock_analysis(state: stock_analyser):
    prompt = PromptTemplate.from_template("""
        You are a Stock Analysis Orchestrator.
        
        User Query:
        {query}
        
        Available agents:
        - income_statement_agent
        - balance_sheet_analysis_agent
        - cashFlowAnalysisAgent
        - technical_agent
        - newsAnalysisAgent
        - PortfolioManagerAgent
        
        Return ONLY valid JSON.
        
        example Output:
        {{
            "agents": [
                "income_statement_agent",
                "technical_agent"
            ]
        }}
        """)
    query = state['context']['query']
    print("heyy")
    print(query)
    print(type(query))
    chain = llm_chains.generatellmchain_withparser(llm_normal,prompt)
    response = chain.invoke({'query':query})
    print(f"RAW LLM RESPONSE: {response!r}")
    agents = response['agents']
    portfolio_response = None
    full_information = state.get('full_information', '')
    for agent in agents:
        internalstate= agent_registry[agent].graph.invoke({'context':state['context']})

        if(agent == 'newsAnalysisAgent'):
            full_information += str(internalstate['news'])
        elif (agent == 'PortfolioManagerAgent'):
            portfolio_response = internalstate['response']
            return {'response': portfolio_response, 'full_information': full_information}
        else:
            full_information += internalstate['analysis']
    if portfolio_response is not None:
        return {'response': portfolio_response, 'full_information': full_information}

    return {'full_information': full_information}

def route_after_stock_analysis(state: stock_analyser):
    if state.get('response'):
        return END
    return "final_conclusion"

def final_conclusion (state: stock_analyser):
    prompt = (f"""Based on the information gathered below, answer the user's query directly and give a clear recommendation — 
            whether to invest, avoid, hold, or whatever action was actually asked for.
            
            User's query: {state['context']['query']}
            
            Information gathered from analysis agents:
            {state['full_information']}
            
            Provide a final, well-reasoned conclusion that directly answers the query above. 
            Be specific and decisive — state your recommendation clearly, then briefly justify it 
            using the most relevant data points from the information provided. do not answer huge""")
    response = llm.llm_normal.invoke(prompt)
    print(response)
    return {'response':response.content}


builder = StateGraph(stock_analyser)
builder.add_node("stock_analysis", stock_analysis)
builder.add_node("final_conclusion", final_conclusion)
builder.add_edge(START, 'stock_analysis')
builder.add_conditional_edges('stock_analysis', route_after_stock_analysis , {"final_conclusion": "final_conclusion", END: END})
builder.add_edge("final_conclusion", END)
print(builder)
graph = builder.compile()

#AI investment research platform