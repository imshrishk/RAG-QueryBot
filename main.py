import pandas as pd
path = r"C:\ParallelDots\Amazon Scraping.xlsx"
dframe = pd.read_excel(path)
data = dframe[["Title", "About This Item"]]
data.head()
data.to_csv("Hitler.txt", sep = "\n", index = False)
from langchain_community.document_loaders import TextLoader
import chardet

# Specify the path to your text file
path = r"C:\ParallelDots\Hitler.txt"

# Detect the encoding of the file
with open(path, 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Create a TextLoader instance with the detected encoding
loader = TextLoader(file_path=path, encoding=encoding)

# Load the documents
docs = loader.load()

# Print the loaded documents
#for doc in docs:
#    print(doc)
#type(docs)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 100)
chunks = text_splitter.split_documents(docs)
#print(chunks[0])
import torch
import sentence_transformers

import torch

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
try:
    from langchain.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma
    print("Libraries imported successfully.")
except ImportError as e:
    print(f"Import error: {e}")
    raise e

# Create a Chroma vector store and add documents
try:
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="rag_chroma", embedding=embeddings)
    print("Vector store created and documents added successfully.")
except Exception as e:
    print(f"Error creating vector store or adding documents: {e}")
    raise e
stored = vectorstore.get()
#print(result)
#print(stored["embeddings"])
stored.keys()
question = """ Recommend me a mascara which has the features listed below: for women can be used over 50, is vitamin enriched, 
easy to remove, has zero lash breakage, is non-clumping and acts as a hypoallergenic for sensitive eyes """
docs = vectorstore.similarity_search(question)
#len(docs)
#for doc in docs:
#    print("********************************************")
#    print(doc)
retriever = vectorstore.as_retriever()
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Prompt
template = """You are an intelligent assistant helping users with their questions as a shopping assistant. 
Use ONLY the following pieces of context to answer the question. Think step-by-step following the given points below and then answer.
Do not try to make up an answer:
- If the context is empty, response with “I do not know the answer to the question.”
- If the answer to the question cannot be determined from the context alone, response with “I cannot determine the answer to the question.”
- If the answer to the question can be determined from the context, response ONLY with "name" where <name> is the product matching the criteria

Question: {question}  
=====================================================================
Context: {context}   
=====================================================================
"""

prompt = ChatPromptTemplate.from_template(template)

# Local LLM
ollama_llm = "gemma"
model_local = ChatOllama(model=ollama_llm)

# Chain
chain = ( {"context": retriever, "question": RunnablePassthrough()}   | prompt   | model_local   | StrOutputParser() )
print(stored["embeddings"])
# Question
chain.invoke(input(""))