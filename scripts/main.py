import time
import re
import pandas as pd
from typing import List
import pdfplumber
import neo4j
import asyncio
from transformers import GPT2Tokenizer
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j import GraphDatabase
from openai import OpenAI
import os


async def main():
    api_key = "YOUR_OPENAI_API_KEY"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"
    # This prompt_template includes our specific structure with elements from the our course syllabus  
    prompt_template = """
    You are an expert in natural language processing. Your task is to extract and structure information from lecture, presenting it in the form of a knowledge graph.
    The knowledge graph should be structured as follows:
    Level 1: Transferable skills, representing foundational knowledge or skills.
    Level 2: Learning outcomes, which define what learners will know, understand, or be able to do upon successful completion of the tasks at Level 5.
    Level 3: Learning tasks, which are the activities required to achieve the learning outcomes at Level 2.
    Level 4: Knowledge, representing the specific knowledge required to perform the tasks at Level 3.
    Level 5: Titles of the lectures within each module.
    Level 6: Names of the modules in the course.
    Level 7: The name of the course.
    
    There are the following types of relationships:
    The relationship types - ‘UNDERSTAND’, ‘APPLY’, ‘ANALYZE’, ‘KNOW’, ‘MAKE’, ‘ASSESS’, ‘CONCEIVE’, ‘DO’, ‘KNOW’, ‘PROPOSE’ describe the transition from the first level to the second level.
    The relationship type ‘FORMER’ describes the transition from level two to level three, from level three to level four, from level four to level five, from level five to level six, and from level six to level seven.
    
    There is an element of entity type ‘REQUEST’ and from it the relationship is ‘FORMED’ to an element of entity type ‘KNOWLEDGE’.
    Each element of entity type ‘LEARNING RESULT’ has a ‘FORMED’ relationship with one or more unique elements of entity type ‘CHALLENGE’.
    Each element of the ‘TASK’ entity type must necessarily have a relationship with one or more elements of the ‘KNOWLEDGE’ entity type.
    Each element of the ‘KNOWLEDGE’ entity type has a ‘FORMAT’ relationship with an element of the ‘LECTURE’ entity type
    
    Build a graph using this description, the graph should contain only these elements:
    From the element ‘UK-1’ of the entity type ‘UNIVERSAL COMPETENCE’ comes the relationship ‘APPLY’ with the element ‘Identify and analyse a problem situation, identifying its structural components and the links between them’ of the entity type ‘LEARNING RESULT’, from this element comes the relationship ‘FORMULATE’ with the element ‘Explore and analyse the possibilities of using Datawrapper service for personal and professional information visualisation purposes’ of the entity type ‘TASK’.
    This element has a ‘FORMER’ relationship with the elements - Visualisation and its possibilities; Modern and perspective visualisation tools; Rules for creating an effective presentation; Overview of popular information visualisation tools of the entity type ‘KNOWLEDGE’, all these elements have a ‘FORMER’ relationship with the element of the entity type ‘LECTURE’
    
    Return the result in JSON format using the following structure:
    {{
      "nodes": [
        {{
          "id": "0",
          "label": "entity type",
          "properties": {{
            "name": "entity name"
          }}
        }}
      ],
      "relationships": [
        {{
          "type": "RELATIONSHIP_TYPE",
          "start_node_id": "0",
          "end_node_id": "1",
          "properties": {{
            "details": "Relationship details"
          }}
        }}
      ]
    }}
    
    Input lecture text:
    
    {text}
    
    Generate only JSON without explanations or additional text
    """
    # Here you can paste your organization rooles
    rooles="""
    1) Formulate the task text in the affirmative.
    2) Place key words in the task text at the beginning of the sentence.
    3) Formulate the task text clearly, concisely, and as briefly as possible, without compromising the meaning.
    4) Exclude verbose reasoning, repetitions, complex syntactic constructions, double negation, and words such as "sometimes," "often," "always," "all," "never."
    5) Avoid using words like "indicate," "select," "list," "name," "all of the above," "all except."
    6) Ensure the absence of spelling, grammatical, and punctuation errors.
    7) The answer options should be meaningful, similar in appearance and grammatical structure, and attractive for selection.
    8) The task formulation should not contain hints to the correct answer.
    9) The text of the answer options should not be too long.
    10) It is recommended to exclude all repetitive words by incorporating them into the main task text.
    11) Answer options such as "all of the above are correct" or "all of the listed answers are incorrect" are not acceptable.
    """

    llm = OpenAILLM(
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1/",
        model_name="Meta-Llama-3.1-405B-Instruct",
        model_params={"temperature": 0.5, "top_p": 0.2}
    )

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    embedder = SentenceTransformerEmbeddings(model="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    builder = GraphBuilder(llm, driver, embedder, prompt_template)
    await builder.build_graph_from_pdf("/content/Module_5_Topic_3.pdf")
    check = Check(data=data, mark='Requirements+Graph', rooles=rooles, driver=driver, llm=llm, embedder=embedder, target_name='Lecture Name')
    df = check.process()
    return df

asyncio.run(main())
