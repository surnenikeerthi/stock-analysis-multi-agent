from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

def generatellmchain(llm, prompt):
    llm_chain = prompt|llm
    return llm_chain

def generatellmchain_withparser(llm,prompt):
    llm_parser_chain = RunnableLambda(lambda _: prompt) | llm | JsonOutputParser()
    return llm_parser_chain