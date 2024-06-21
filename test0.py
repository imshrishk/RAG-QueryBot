from typing import List
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
    DocumentSummaryIndex,
    get_response_synthesizer,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv
import openai
import os
#import faiss
import pandas as pd
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# d = 1024
# faiss_index = faiss.IndexFlatL2(d)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("shopping")

data_directory = r"C:\ParallelDots"

class ExcelFileReader(BaseReader):
    def load_data(self, file, extra_info=None) -> List[Document]:
        df = pd.read_excel(file)
        text = df.to_string(index=False)
        return [Document(text=text + " Foobar", extra_info=extra_info or {})]

file_extractor = {".xlsx": ExcelFileReader()}
documents = SimpleDirectoryReader(input_dir=data_directory, filename_as_id=True, file_extractor=file_extractor).load_data()
Settings.llm = Ollama(model="llama3:instruct", request_timeout=1000)
llm = Settings.llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# embed_model = Settings.embed_model

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
splitter = SentenceSplitter(chunk_size=1500, chunk_overlap = 100)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("shopping")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,show_progress=True, transformations=[splitter])

index.storage_context.persist(persist_dir="./storage")

db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("shopping")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    transformations=[splitter]
)

qa_prompt_str='''Recommend me a eyeshadow stick which has the features listed: stick set 8 color, Multi-Purpose Makeup Eyeshadow Set: bold, creamy, crease resistant color with ease, durable and waterproof without wrinkles, smooth as silk without lumps, does not fade and has uniform color, can be used as a contouring or Highlight, has 8 colors , Easy to Use, No brushes required, just glide, highly pigmented, retractable eyeshadow stick for last part, multi -combination eyeshadow, Classy & Buildable Base Color, its soft and creamy texture  keep it subtle for a natural look!'''

# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=4,
# )

# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
# )

query_engine = index.as_query_engine(response_synthesizer=response_synthesizer, llm=llm)

response = query_engine.query(qa_prompt_str)

print(response)
