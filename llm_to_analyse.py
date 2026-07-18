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
    decision = llm_chain.invoke({'query': query}).content
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
Extract the stock/company name and its Yahoo Finance ticker symbol from the following user query.

User Query:
{query}

Return ONLY valid JSON in the following format:

{{
    "stock_name": "<company name>",
    "ticker": "<Yahoo Finance ticker symbol>"
}}

The ticker MUST include the appropriate Indian stock exchange suffix:
- Use ".NS" for stocks listed on the National Stock Exchange (NSE).
- Use ".BO" for stocks listed on the Bombay Stock Exchange (BSE).
- Prefer the ".NS" ticker when the stock is listed on both exchanges.

Examples:
- Tata Consultancy Services → "TCS.NS"
- Infosys → "INFY.NS"
- Reliance Industries → "RELIANCE.NS"

If the company or ticker cannot be identified, return:

{{
    "stock_name": null,
    "ticker": null
}}

Return ONLY the JSON object. Do not include any explanation, markdown, or additional text.
""")
    chain = llm_chains.generatellmchain(llm.llm_normal, prompt)
    response = chain.invoke({'query' : state['messages'][-1].content})
    json_response = json.loads(response.content)
    stock_name = json_response['stock_name']
    ticker = json_response['ticker']
    return {'stock_name' : stock_name, 'ticker':ticker}

def call_stock_agent(state: prompt_orchestrator):
    response = stock_analyser.graph.invoke({'context':{'stock_name' : state['stock_name'], 'ticker':state['ticker'], 'query' : state['messages'][-1].content},'stock_financial' : '',
    'balance_sheet' : '', 'response' :'', 'full_information':''})
    print("did u come out")
    print(response.keys())
    final_response = response['response']
    print(final_response)
    return {"messages" : [final_response]}

def reject_node(state:prompt_orchestrator):
    #call another llm that will judge whether to actually reject or can answer
    value = "sorry , I cannot help you with that query I am limited to stock Analysis and managment"
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