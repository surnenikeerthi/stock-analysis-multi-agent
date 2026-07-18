from langchain_openai import ChatOpenAI
llm_normal=ChatOpenAI(model = "gpt-4o-mini")
llm_critic=ChatOpenAI(model = "gpt-4")