from agent.nodes import router_node

for q in [
    "How much is the individual deductible?",
    "What does coinsurance mean in general?",
    "Review the whole policy and find every problem you can find.",
    "Why shall I take insurance?",
]:
    decision = router_node({"query": q, "conversation_history": []})
    print(f"{decision['route']} -> {q}")