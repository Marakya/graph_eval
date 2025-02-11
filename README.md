# Approach to Evaluating AI-Generated Educational Content: A Case Study on AI-Generated Tests 
![visualisation (2)](https://github.com/user-attachments/assets/af67eb44-48b9-44fb-a681-6b6dab20db8d)

An approach for evaluating learning materials using the knowledge graph is presented here.


We propose an approach to automatically [generate a knowledge graph](https://github.com/Marakya/graph_eval/blob/main/scripts/build_graph.py) and save it to a graph database - [Neo4j](https://neo4j.com/)

Also subsequent [evaluation](https://github.com/Marakya/graph_eval/blob/main/scripts/check_evaluation.py) of educational materials (tests as an example), using LLM+GraphRAG - which allows extracting relevant nodes from the graph and feeding them into a language model.
