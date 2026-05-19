import os
import shutil
import time
from typing import List, Optional

from .base import PredictionResult, RAGSystem

from llama_index.core import (
    Settings,
    StorageContext,
    Document,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


class ReverseOrderPostprocessor(BaseNodePostprocessor):
    """Places the most relevant chunk last, just before the query (mitigates lost-in-the-middle)."""

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        return list(reversed(nodes))


class OnePassRAG(RAGSystem):

    settings_initialized = False

    similarity_top_k = 12
    chunk_size = 512
    chunk_overlap = 128  # ~25% overlap approximates sliding window (paper Table 4)

    text_qa_template = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Using only the context above, answer the question concisely. "
        "If the answer is not in the context, output only the word: unanswerable\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )

    refine_template = PromptTemplate(
        "The original question is: {query_str}\n"
        "Existing answer: {existing_answer}\n"
        "Additional context is below.\n"
        "---------------------\n"
        "{context_msg}\n"
        "---------------------\n"
        "Refine the existing answer using the additional context if it is helpful. "
        "Otherwise keep the existing answer. "
        "If the answer is not in the context, output only the word: unanswerable\n"
        "Refined answer:"
    )

    def setup(self, document_text: str) -> None:
        if not OnePassRAG.settings_initialized:
            # Settings.embed_model = HuggingFaceEmbedding(
            #     model_name="BAAI/bge-small-en-v1.5",
            #     device="cuda",
            # )
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/llm-embedder",
                device="cuda",
                query_instruction="Represent this question for searching relevant passages: ",
                text_instruction="Represent this passage for retrieval: ",
            )
            # TODO: Ollama using quantized model, maybe try with vLLM backend since it uses BF16.
            Settings.llm = Ollama(model="llama3.1", request_timeout=200.0)
            OnePassRAG.settings_initialized = True

        # Setup a local directory for the database
        self.db_uri = "./lancedb"
        if os.path.exists(self.db_uri):
            shutil.rmtree(self.db_uri)  # Ensure a clean slate

        # Create vector store
        vector_store = LanceDBVectorStore(
            uri=self.db_uri,
            table_name="niah_table",
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Ingest document with smaller chunks to reduce retrieval noise.
        text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        doc = Document(text=document_text)
        self.index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
            transformations=[text_splitter],
        )

    def predict(self, query: str) -> PredictionResult:
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            response_mode="compact",
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
            node_postprocessors=[ReverseOrderPostprocessor()], # better perfoming repacking ordering
        )

        start_time = time.perf_counter()
        response = query_engine.query(query)
        end_time = time.perf_counter()

        # Get retrieved context
        retrieved_context = []
        for source_node in response.source_nodes:
            # source_node is a wrapper that contains the chunk and its similarity score
            chunk_text = source_node.node.text
            score = source_node.score

            retrieved_context.append((chunk_text, score))

        return PredictionResult(
            answer=response.response,
            execution_time_seconds=end_time - start_time,
            retrieved_context=retrieved_context,
        )

    def teardown(self) -> None:
        shutil.rmtree(self.db_uri)
