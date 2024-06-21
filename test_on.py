from typing import List
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv
import openai
import os
import pandas as pd
from llama_index.core.query_engine import RetrieverQueryEngine
from IPython.display import Markdown, display

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB persistent client
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("shopping")

# Data directory
data_directory = r"C:\ParallelDots"

# Custom Excel reader
class ExcelFileReader(BaseReader):
    def load_data(self, file, extra_info=None) -> List[Document]:
        df = pd.read_excel(file)
        text = df.to_string(index=False)
        return [Document(text=text + " Foobar", extra_info=extra_info or {})]

# Read documents from directory
file_extractor = {".xlsx": ExcelFileReader()}
documents = SimpleDirectoryReader(input_dir=data_directory, filename_as_id=True, file_extractor=file_extractor).load_data()

# Set up the LLM and embedding model
Settings.llm = Ollama(model="gemma:instruct", request_timeout=1000, temperature = 0.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)

# Sentence splitter
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

# Vector store
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("shopping")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True, transformations=[splitter]
)

# # Persist storage context
# index.storage_context.persist(persist_dir="./storage")

# Re-initialize ChromaDB client and collection
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("shopping")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load index from vector store
index = VectorStoreIndex.from_vector_store(
    vector_store,
    transformations=[splitter],
)

# Query engine
query_engine = index.as_query_engine(response_mode="compact", similarity_top_k=2)

# Define QA prompt template
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "You are an intelligent assistant helping users with their questions as a shopping assistant. Think step-by-step following the given points below and then answer.\n"
    " 1. If the context is empty, response with “I do not know the answer to the question. \n"
    " 2. If the answer to the question cannot be determined from the context alone, response with \“I cannot determine the answer to the question.\"\n"
    " 3. If the answer to the question can be determined from the context, response ONLY with \"name\" where <name> is the product matching the criteria.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)


# Update prompts in query engine
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

# # Function to display prompt dictionary
# def display_prompt_dict(prompts_dict):
#     for k, p in prompts_dict.items():
#         text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
#         display(Markdown(text_md))
#         print(p.get_template())
#         display(Markdown("<br><br>"))

# # Display prompts
# prompts_dict = query_engine.get_prompts()
# display_prompt_dict(prompts_dict)

# Display
qa_prompt_str = ''' Recommend me a eyeshadow stick set which has the features listed below:  2 classic colors can be used to create smoky,sultry,luminous,metallic finish,neutral,and luminous eye, works well as a highlighter and Eyeliner Pencil, Easy to use, Designed for smooth blending and even color payoff, Goes on smoothly like silk without clumping,skipping or smudging,perfect for sculpting, shading & defining,not exaggerated glittery, Powder Waterproof & Long Lasting, Long lasting, stay on & Up to 10~12 hours wear, Perfect Packaging & Nice Gift  '''

response = query_engine.query(qa_prompt_str)
print(response)
