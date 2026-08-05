from agent.nodes import filter_node

for q in ["What is the maternity waiting period?", "Review the entire policy."]:
    mf = filter_node({"query": q})["metadata_filter"]
    print(f"{q}\n -> sections: {mf.section}\n")