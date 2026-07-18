from typing import TypedDict,Any
from langchain_core.prompts import PromptTemplate
import yfinance as yf
import llm, llm_chains
import json
from langgraph.graph import StateGraph,  START, END
class newsAnalysisAgent(TypedDict):
    context :Any
    news: Any
    full_news : Any
    analysis :Any

def get_news_from_yfinance(state:newsAnalysisAgent):
    ticker_obj = yf.Ticker(state['context']['ticker'])
    news_full = ticker_obj.news
    news = fetch_news_from_full(news_full)
    return {'full_news':news_full, 'news':news}

def fetch_news_from_full(news_full):
    top_5_news = []
    for i in range(10):
        if(i<6):
            top_5_news.append(news_full[i]['content']['summary'])
        else:
            top_5_news.append(news_full[i]['content']['title'])
    return top_5_news

def news_fetch_based_on_header(header, news_full):
    extra = []
    for news in news_full:
        title = news.get('content', {}).get('title')
        summary = news.get('content', {}).get('summary')
        if title in header:
            extra.append(summary)
    return extra

def llm_validate_news(state:newsAnalysisAgent):
    prompt = (f"""These five summaries describe the most recent events.

The remaining ten articles are represented only by their titles.

Determine whether any of these titles appear significant enough that their full summaries should also be included before performing investment analysis.
input : 5 summaries and 5 titles:
{state['news']}
Return ONLY JSON:

{{
    "need_more_context": true,
    "titles": [...]
}}""" )
    news = state['news']
    chain = llm_chains.generatellmchain_withparser(llm.llm_normal, prompt)
    response = chain.invoke({'news': news})
    if (response['need_more_context']):
        extra_news = news_fetch_based_on_header(response['titles'], state['full_news'])
    #response = graph.invoke({'news': news})

    news = news[:5] + extra_news
    return {'news': news}

builder = StateGraph(newsAnalysisAgent)
builder.add_node("get_news_from_yfinance", get_news_from_yfinance)
builder.add_node("llm_validate_news", llm_validate_news)
builder.add_edge(START, "get_news_from_yfinance")
builder.add_edge("get_news_from_yfinance",  "llm_validate_news")
builder.add_edge("llm_validate_news", END)

graph = builder.compile()