from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

def generatellmchain(llm, prompt):
    llm_chain = prompt|llm
    return llm_chain

def generatellmchain_withparser(llm,prompt):
    llm_parser_chain = RunnableLambda(lambda _: prompt) | llm | JsonOutputParser()
    return llm_parser_chain

def generatellm_with_tools(llm, prompt, tools):
    llm_tools = llm.bind_tools(tools)
    llm_with_tools =  prompt|llm_tools
    return llm_with_tools