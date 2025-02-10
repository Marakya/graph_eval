class GraphBuilder:
    def __init__(self, llm, driver, embedder, prompt_template):
        self.llm = llm
        self.driver = driver
        self.embedder = embedder

        self.node_labels = ['COURSE', 'MODULE', 'LECTURE', 'KNOWLEDGE', 'LEARNING_OUTCOME', 'TASK', 'UNIVERSAL_COMPETENCE']
        self.rel_types = ["UNDERSTAND", "APPLY", "ANALYZE", "BE_ABLE_TO", "EVALUATE", "CREATE", "DO", "KNOW", "SUGGEST", "CONTAINS", "FORMS"]

        self.text_splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=50)
        self.prompt_template = prompt_template

        self.kg_builder = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            text_splitter=self.text_splitter,
            embedder=self.embedder,
            entities=self.node_labels,
            relations=self.rel_types,
            prompt_template=self.prompt_template,
            from_pdf=False
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF document."""
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        return all_text

    def split_into_chunks_with_overlap(self, text, max_tokens=2000, overlap=50):
        """Splits text into chunks with token overlap."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))

            while end > start and not self.tokenizer.decode(tokens[start:end]).endswith(" "):
                end -= 1

            chunk_text = self.tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - overlap if end - overlap > start else end

        return chunks

    async def build_graph_from_pdf(self, pdf_path):
        """Extracts text from a PDF, splits it into chunks, and builds a knowledge graph."""
        print(f"Extracting text from {pdf_path}...")
        pdf_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks_with_overlap(pdf_text)

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}...")
            result = await asyncio.to_thread(self.kg_builder.run, text=chunk)
            print(f"Result: {result}")
            await asyncio.sleep(20)
