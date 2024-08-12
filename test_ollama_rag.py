# Importing the libraries
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Importing the xlsx file 
path = "/home/haruki/Desktop/VS_Code/ollama/AmazonProducts.xlsx"
dframe = pd.read_excel(path)

# Preparing the data
dframe = dframe[dframe['About This Item'].apply(lambda x: len(x) > 2)]
dframe = dframe.drop_duplicates()

# Getting the related content
data = dframe[["Title", "About This Item"]]
data.head()
# Storing the relevant data into text file
data.to_csv("amazon.txt", sep = "\n", index = False)

# Importing the stroed data as document
path = "/home/haruki/Desktop/VS_Code/ollama/data/amazon.txt"
loader = TextLoader(path)
docs = loader.load()

# divinding the given data into chunks to store into vector database(CHROMADB)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1700, chunk_overlap = 100)
chunks = text_splitter.split_documents(docs)
# Checking the chunk
print(chunks[0])

# Index 
model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name = model)

# Storing the chunks in vector store
vectorstore = Chroma.from_documents( documents = chunks, collection_name = "rag_chroma", embedding = embeddings, )
retriever = vectorstore.as_retriever()

# checking the vector store similarity search
question = """ Eyeshadow stick which has the features listed below:  stick set 8 color, 
Multi-Purpose Makeup Eyeshadow Set: bold, creamy, crease resistant color with ease, durable and 
waterproof without wrinkles, smooth as silk without lumps, does not fade and has uniform color, can be used as a 
contouring or Highlight, has 8 colors , Easy to Use, No brushes required, just glide, highly pigmented, 
retractable eyeshadow stick for last part, multi -combination eyeshadow, Classy & Buildable Base Color, its soft and 
creamy texture  keep it subtle for a natural look! """
docs = vectorstore.similarity_search(question)
print(len(docs))

# Prompt
template = """  You are an intelligent assistant helping users with their questions as a shopping assistant. 
Use ONLY the following pieces of context to answer the question. Think step-by-step following the given points below and then answer.
Do not try to make up an answer:
- Check the numbers given in the question to answer the names of the product matching the criteria, and give names accordingly
- If the context is empty, response with “I do not know the answer to the question.”
- If the answer to the question cannot be determined from the context alone, response with “I cannot determine the answer to the question.”
- If the answer to the question can be determined from the context, response ONLY with "name and color" where <name> is the product 
matching the criteria and the color it's available in.

Question: {question}  
=====================================================================
Context: {context}   
=====================================================================
"""

prompt = ChatPromptTemplate.from_template(template)

# Local LLM
model_local = ChatOllama(model = "gemma")

# Chain
chain = ( {"context": retriever, "question": RunnablePassthrough()}   | prompt   | model_local   | StrOutputParser() )

answer = chain.invoke(input(""))

print(answer)
