from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph,START,END, MessagesState
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict
import llm_chains
import json
import stock_analyser
from langgraph.checkpoint.memory import InMemorySaver
import llm
checkpointer= InMemorySaver()
class prompt_orchestrator(MessagesState):
    decision: str
    stock_name: str
    ticker: str
    bought_price : float
    stock_count : int
    total_amount : float

def prompt_validation(state : prompt_orchestrator):
    query = state['messages'][-1].content
    print(state['messages'])
    prompt = PromptTemplate.from_template("You are a query classifier for a stock analysis assistant."
            "Classify the user's query as:"
            
            "- stock: related to stocks, investing, markets, company financials, trading, or portfolios."
            "- general: greetings or simple conversational questions (e.g. hi, hello, how are you, what can you do)."
            "- reject: The query is unrelated to stock analysis and is outside the scope of this assistant."
            "Examples:"
            "Book me a bus ticket."
            "Write a Python game."
            
            "Return exactly one lowercase word:"
            "stock"
            "general"
            "reject"                              
            
            "Query: {query}")
    llm_chain = llm_chains.generatellmchain(llm.llm_normal, prompt)
    decision = llm_chain.invoke({'query': state['messages']}).content
    print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    print(decision)
    return {'decision': decision}

def decision_maker(state:prompt_orchestrator):
    decision = state['decision']
    if decision == 'stock':
        query = state['messages'][-1].content
        classifier_label = stock_query_critic(query)
        print(classifier_label)
        return classifier_label
    if decision == 'general':
        return 'respond2genprompt'
    if decision == 'reject':
        return 'reject_node'

def stock_query_critic(query):
    prompt = (f"""You are a classifier that decides how to handle a stock-related query.
            
            Determine whether answering the query requires performing stock analysis or 
            calling one of the analysis agents, or whether it is a generic, stock-related 
            question that can be answered directly.
            
            Return exactly one of the following:
            - "respond2genprompt" — if the query is stock-related but generic (conceptual, 
              educational, or definitional; no specific stock, portfolio action, or data lookup needed)
            - "stock_analysis" — if the query requires performing an action, such as fetching 
              news, analyzing technical data, evaluating a specific stock, or managing/checking a portfolio
            
            Examples:
            - "What is RSI?" → generic, no specific action needed → respond2genprompt
            - "Should I invest in TCS?" → requires analysis of a specific stock → stock_analysis
            - "What is the recent news on Infosys?" → requires fetching data → stock_analysis
            - "List the stocks I have invested in?" → requires querying the portfolio → stock_analysis
            
            Return only the classification label, nothing else.
            here is the question : {query}""" )
    response = llm.llm_critic.invoke(prompt)
    return response.content

def stock_analysis(state:prompt_orchestrator):
    prompt = PromptTemplate.from_template("""
Extract the stock/company name, its Yahoo Finance ticker symbol, and any stock purchase details explicitly mentioned in the following user query.

User Query:
{query}

Return ONLY valid JSON in the following format:

{{
    "stock_name": "<company name or null>",
    "ticker": "<Yahoo Finance ticker symbol or null>",
    "bought_price": <price per stock or null>,
    "stock_count": <number of stocks/shares or null>,
    "total_amount": <total purchase amount or null>
}}

### Stock/Company Extraction Rules

1. Identify the company/stock name mentioned in the user query.
2. Identify the corresponding Yahoo Finance ticker symbol.
3. The ticker MUST include the appropriate Indian stock exchange suffix:
   - Use ".NS" for stocks listed on the National Stock Exchange (NSE).
   - Use ".BO" for stocks listed on the Bombay Stock Exchange (BSE).
   - Prefer the ".NS" ticker when the stock is listed on both exchanges.

Examples:
- Tata Consultancy Services → "TCS.NS"
- Infosys → "INFY.NS"
- Reliance Industries → "RELIANCE.NS"

If the company or ticker cannot be identified, return null for the corresponding field.

### Purchase Details Extraction Rules

Extract the following values ONLY when they are explicitly mentioned or can be directly determined from the user's query:

1. "bought_price"
   - The price paid per stock/share.
   - Examples:
     - "I bought Infosys at 1500" → 1500
     - "bought 10 shares at ₹1,500 each" → 1500
   - Do not confuse the current market price with the bought price.

2. "stock_count"
   - The number of shares/stocks purchased.
   - Examples:
     - "I bought 10 shares" → 10
     - "I own 25 Reliance shares" → 25

3. "total_amount"
   - The total amount invested/purchased.
   - Examples:
     - "I invested ₹50,000 in Infosys" → 50000
     - "Bought 10 shares at ₹1,500 each" → 15000
   - If the total amount is explicitly stated, use that value.
   - If total amount is not explicitly stated but can be directly calculated from bought_price × stock_count, calculate and return it.
   - Do not calculate total_amount using the current market price.

### Important Rules

- Return null for any field that is not mentioned and cannot be directly determined.
- Do not invent or assume purchase details.
- If bought_price, stock_count, and total_amount are all available, return all three.
- If only one or two purchase details are available, return the available values and null for the others.
- Numeric values must be returned as JSON numbers, not strings.
- Remove currency symbols such as ₹, $, or commas from numeric values.
  - ₹1,50,000 → 150000
  - ₹2,500 → 2500
- If the user mentions multiple stocks, extract the stock/company that the query is primarily referring to. Do not return multiple JSON objects.
- Preserve null values as JSON null.

### Examples

User Query:
"I bought 10 Infosys shares at ₹1,500 each."

Output:
{{
    "stock_name": "Infosys",
    "ticker": "INFY.NS",
    "bought_price": 1500,
    "stock_count": 10,
    "total_amount": 15000
}}

User Query:
"I invested ₹1,00,000 in TCS."

Output:
{{
    "stock_name": "Tata Consultancy Services",
    "ticker": "TCS.NS",
    "bought_price": null,
    "stock_count": null,
    "total_amount": 100000
}}

User Query:
"Should I buy Infosys for the long term?"

Output:
{{
    "stock_name": "Infosys",
    "ticker": "INFY.NS",
    "bought_price": null,
    "stock_count": null,
    "total_amount": null
}}


Return ONLY the JSON object. Do not include any explanation, markdown, or additional text.
""")
    chain = llm_chains.generatellmchain(llm.llm_normal, prompt)
    response = chain.invoke({'query' : state['messages'][-1].content})
    json_response = json.loads(response.content)
    stock_name = json_response['stock_name']
    ticker = json_response['ticker']
    total_amount =json_response['total_amount']
    bought_price = json_response['bought_price']
    stock_count = json_response['stock_count']
    return {'stock_name' : stock_name, 'ticker':ticker, 'bought_price':bought_price, 'stock_count':stock_count,'total_amount':total_amount}

def call_stock_agent(state: prompt_orchestrator):
    response = stock_analyser.graph.invoke({'context':{'stock_name' : state['stock_name'], 'ticker':state['ticker'], 'query' : state['messages'][-1].content, 'bought_price':state['bought_price'], 'stock_count':state['stock_count'],'total_amount':state['total_amount']},'stock_financial' : '',
    'balance_sheet' : '', 'response' :'', 'full_information':''})
    print("did u come out")
    print(response.keys())
    final_response = response['response']
    print(final_response)
    return {"messages" : [final_response]}

def reject_node(state:prompt_orchestrator):
    #call another llm that will judge whether to actually reject or can answer
    value = "sorry , I cannot help you with that query I am limited to stock Analysis and management"
    prompt = PromptTemplate.from_template("You are a critic for a stock analysis assistant."
            "Return:"
            "- allow: if the query is stock-related or a simple conversational/general question."
            "- reject: if the query is unrelated to the assistant's purpose."
            
            "Return exactly one lowercase word: allow or reject."
            
            "Query: {query}")
    chain = llm_chains.generatellmchain(llm.llm_critic, prompt)
    response = chain.invoke({'query': state['messages'][-1].content}).content
    if(response == 'allow'):
        value= llm.llm_normal.invoke(state['messages'][-1].content)

    return {"messages" : [value]}

def respond2genprompt(state:prompt_orchestrator):
    query = state['messages'][-1].content
    # system message to be added here in the prompt
    response = llm.llm_normal.invoke(state["messages"] + [HumanMessage(content=query)])
    return {"messages": [response]}

builder = StateGraph(prompt_orchestrator)
builder.add_node("prompt_validation", prompt_validation)
builder.add_node("respond2genprompt", respond2genprompt)
builder.add_node("reject_node", reject_node)
builder.add_node("stock_analysis", stock_analysis)
builder.add_node("call_stock_agent", call_stock_agent)
builder.add_edge(START, "prompt_validation")
builder.add_conditional_edges("prompt_validation",decision_maker)
builder.add_edge("respond2genprompt", END)
builder.add_edge("reject_node", END)
builder.add_edge("stock_analysis", "call_stock_agent")
builder.add_edge("call_stock_agent", END)
print(builder)
graph = builder.compile(checkpointer = checkpointer)