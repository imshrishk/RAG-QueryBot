# Importing the libraries
import shelve
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Similarity search of the vector store itself
def SimilaritySearch(vectorstore, question):
    docs = vectorstore.similarity_search(question)
    return docs


# Searching the products using GEMMA(Ollama model)
def productSearch(vectorstore, question):
    retriever = vectorstore.as_retriever()
    # Prompt
    template = """  You are an intelligent assistant helping users with their questions as a shopping assistant. 
    Use ONLY the following pieces of context to answer the question. Think step-by-step following the given points below and then answer.
    Do not try to make up an answer:
    - Check the numbers given in the question to answer the names of the product matching the criteria, and give names accordingly
    - If the context is empty, response with “I do not know the answer to the question.”
    - If the answer to the question cannot be determined from the context alone, response with “I cannot determine the answer to the question.”
    - If the answer to the question can be determined from the context, response ONLY with "name" where <name> is the product 
    matching the criteria.

    Question: {question}  
    =====================================================================
    Context: {context}   
    =====================================================================
    """

    prompt = ChatPromptTemplate.from_template(template)
    # 
    model = ChatOllama(model = "gemma")
    # Chain
    chain = ( {"context": retriever, "question": RunnablePassthrough()}   | prompt   | model   | StrOutputParser() )
    answer = chain.invoke(question)

    # Shelve database for retrieval
    shelv = shelve.open("/home/haruki/Desktop/VS_Code/ParallelDots/PD_Beauty")
    names = answer.split(",")
    
    # Storing the retrieved data
    result = {}
    shelv_dict = dict(shelv)
    for name in names:
        for key in shelv_dict:
            if name.lower() in key.lower():
                if key not in result.keys():
                    result[key] = shelv_dict[key]


    return result


if __name__ == "__main__":

    # Loading the vectorstore
    @st.cache_resource
    def get_vectorstore():
        model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name = model)

        path = "/home/haruki/Desktop/VS_Code/ParallelDots/VectorStore"
        vectorstore = Chroma( collection_name="PD_BeautyProducts", embedding_function = embeddings,  
                        persist_directory = path)

        return vectorstore

    vectorstore = get_vectorstore()

    # Query
    st.subheader("Enter your query to search product")
    question = st.text_area("Enter your query")

    # Output
    if len(question) >  10:
        names_dict = productSearch(vectorstore, question)

        if names_dict:        
            for key, value in names_dict.items():
                parts = value.split("\n")
                link, description = parts[0], parts[1] if len(parts) > 1 else ""

                st.success(key)
                st.write("Buy: ", link)
                st.write("Description: ", description)

        else:
            st.error("No results found")