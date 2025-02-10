class Check:
    def __init__(self, data, mark, rooles, driver, llm, embedder, target_name):
        self.__data = data
        self.__mark = mark
        self.__rooles= rooles
        self.driver = driver
        self.llm = llm
        self.embedder = embedder
        self.target_name = target_name

    def check_with_graph_rules(self, tests: List[str], rooles: str, extra: str) -> str:
        q = f"""
        You are a teacher who needs to evaluate test questions. You have an initial assessment of the test questions and additional knowledge about them.
        Revise the assessment according to the additional knowledge. You can change the assessment only in the following case:
        - If the initial assessment is 1, you can change it to 0 if the test question does not cover the additional knowledge.

        Response structure:
        | Question | Compliance with requirements |
        | --- | --- |

        Test questions:
        {tests}

        Initial assessment:
        {rooles}

        Additional knowledge:
        {extra}
        """
        return self.llm.invoke(q).content

    def check_with_graph(self, tests: List[str], extra: str) -> str:
        q = f"""
        You are a teacher who needs to evaluate test questions. You have additional knowledge about these questions.
        Rate the test questions according to the additional knowledge.
        Provide the response as a table with questions and a column with 0 and 1, where 0 means the test question does not comply with the additional knowledge, and 1 means it does.

        Response structure:
        | Question | Compliance with requirements |
        | --- | --- |

        Test questions:
        {tests}

        Additional knowledge:
        {extra}
        """
        return self.llm.invoke(q).content

    def check(self, tests: List[str], rooles: str) -> str:
        q = f"""
        You are a teacher who needs to evaluate test questions.
        Assess the test questions for compliance with university requirements.
        Provide the response as a table with questions and a column with 0 and 1, where 0 means the test question does not comply with the requirements, and 1 means it does.

        Response structure:
        | Question | Compliance with requirements |
        | --- | --- |

        Test questions:
        {tests}

        University requirements:
        {rooles}
        """
        return self.llm.invoke(q).content

    def extract_chunks(self) -> List[str]:
        create_vector_index(driver, name="text_embeddings", label="__Entity__",
                            embedding_property="embedding", dimensions=1536, similarity_fn="cosine")

        vector_retriever = VectorRetriever(
            self.driver, index_name="text_embeddings", embedder=self.embedder, return_properties=["text"]
        )
        vector_res = vector_retriever.get_search_results(query_text=self.target_name, top_k=5)
        return [record.data()["node"]["text"] for record in vector_res.records]

    def split_doc(self, doc_full: str) -> List[str]:
        doc_parts = []
        while len(doc_full) > 10000:
            last_space_index = doc_full[:10000].rfind(' ')
            if last_space_index != -1:
                doc_parts.append(doc_full[:last_space_index])
                doc_full = doc_full[last_space_index:].strip()
            else:
                doc_parts.append(doc_full[:10000])
                doc_full = doc_full[10000:].strip()
        if doc_full:
            doc_parts.append(doc_full)
        return doc_parts

    def process(self):
        tests = self.__data['AI'][self.__data['Lecture'] == self.target_name].tolist()

        graph_data = self.retrieve_graph()
        graph_summary = self.summarize_graph(graph_data)

        if self.__mark == 'Requirements':
            answer_llm = self.check(tests, self.__rooles)
        elif self.__mark == 'Requirements+Graph':
            answer_llm = self.check_with_graph_rules(tests, self.__rooles, graph_summary)
        elif self.__mark == 'Graph':
            answer_llm = self.check_with_graph(tests, graph_summary)
        else:
            raise ValueError("Unsupported marking type")

        return self.parse_response(answer_llm)

    def retrieve_graph(self):
        query = f"""
            MATCH (e1:__Entity__)-[r*]->(target:__Entity__ {{name: '{self.target_name}'}})
            RETURN e1, r, target;
        """
        with self.driver.session() as session:
            return [str(record) for record in session.run(query)]

    def summarize_graph(self, graph_data: List[str]) -> str:
        graph_chunks = self.split_doc(" ".join(graph_data))
        summaries = [self.llm.invoke(f"There is a graph {chunk}. Summarize its key nodes and relationships.").content for chunk in graph_chunks]
        return " ".join(summaries)

    def parse_response(self, response: str) -> pd.DataFrame:
        matches = re.findall(r'\|\s*(.*?)\s*\|\s*(\d+)\s*\|', response)
        return pd.DataFrame(matches, columns=["Question", "Compliance with requirements"])
