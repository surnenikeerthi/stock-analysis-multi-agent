import PortfolioManagerAgent
print("hii")
result = PortfolioManagerAgent.graph.invoke({
    "messages": [],
    "context": {"query": "Add 10 shares of INFY.NS at 1500 rupees today"}
})
print(result.get("response"))