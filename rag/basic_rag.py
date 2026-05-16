from ABCs import BaselineSystem, PredictionResult

import os
import shutil
import time

import lancedb
from llama_index.core import Settings, StorageContext, Document, VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

class BasicRAG(BaselineSystem):

    settings_initialized = False


    def setup(self, document_text: str) -> None:
        if not BasicRAG.settings_initialized:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                device="cuda" 
            )
            Settings.llm = Ollama(model="llama3", request_timeout=200.0)
            BasicRAG.settings_initialized = True
            
        # Setup a local directory for the database
        self.db_uri = "./lancedb"
        if os.path.exists(self.db_uri):
            shutil.rmtree(self.db_uri) # Ensure a clean slate
        
        # Initialize LanceDB
        db = lancedb.connect(self.db_uri)

        # Create a table
        # We define the vector store via LlamaIndex
        vector_store = LanceDBVectorStore(
            uri=self.db_uri, 
            table_name="niah_table"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Ingest Document
        doc = Document(text=document_text)
        self.index = VectorStoreIndex.from_documents(
            [doc], 
            storage_context=storage_context
        )


    def predict(self, query: str) -> PredictionResult:
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        start_time = time.perf_counter()
        response = query_engine.query(query)
        end_time = time.perf_counter()
        
        return PredictionResult(answer=response.response, 
                                  execution_time_seconds=end_time-start_time)


    def teardown(self) -> None:
        shutil.rmtree(self.db_uri)
