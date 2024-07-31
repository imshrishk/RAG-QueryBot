import streamlit as st
import pandas as pd
import chardet
import shelve
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_and_prepare_data(excel_path, text_path, chunk_size, chunk_overlap, cache_path='data_cache.db'):
    with shelve.open(cache_path) as cache:
        if 'chunks' in cache:
            return cache['chunks']
        
        # Load Excel data
        dframe = pd.read_excel(excel_path)
        data = dframe[["url", "title", "about_this_item"]]
        data.to_csv(text_path, sep="\n", index=False)

        # Detect the encoding of the file
        with open(text_path, 'rb') as file:
            result = chardet.detect(file.read())
            encoding = result['encoding']
        
        # Load text documents
        loader = TextLoader(file_path=text_path, encoding=encoding)
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)

        # Cache the chunks
        cache['chunks'] = chunks

    return chunks

def create_vector_store(chunks):
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Chroma vector store and add documents
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="rag_chroma", embedding=embeddings)
    return vectorstore

def visualize_data(dframe):
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    sns.countplot(data=dframe, y='title', ax=ax)
    st.pyplot(fig)

def main():
    st.title("Advanced Shopping Assistant")

    # Sidebar for user adjustments
    st.sidebar.header("Settings")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=1000, value=400, step=50)

    # File upload for Excel data
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is not None:
        excel_path = uploaded_file.name
        text_path = "data.txt"

        try:
            # Load and prepare data
            chunks = load_and_prepare_data(uploaded_file, text_path, chunk_size, chunk_overlap)
            vectorstore = create_vector_store(chunks)

            # Display vector store info
            st.write("Vector store created and documents added successfully.")

            # Show raw data and chunks for debugging
            dframe = pd.read_excel(uploaded_file)
            with st.expander("Show Raw Data"):
                st.write(dframe)
            with st.expander("Show Document Chunks"):
                st.write(chunks)

            # Data Visualization
            visualize_data(dframe)

            # Search history
            if 'history' not in st.session_state:
                st.session_state.history = []

            # User input for question
            question = st.text_input("Ask a question about the products:")
            if question:
                docs = vectorstore.similarity_search(question)
                retriever = vectorstore.as_retriever()

                # Define the prompt
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
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | model_local
                    | StrOutputParser()
                )

                # Get the answer
                answer = chain.invoke(question)
                st.write("Answer:", answer)

                # Store search history
                st.session_state.history.append((question, answer))

            # Display search history
            with st.expander("Search History"):
                for q, a in st.session_state.history:
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:** {a}")
                    st.write("---")

            # User feedback
            feedback = st.text_input("Provide feedback on the answer:")
            if feedback:
                st.write("Thank you for your feedback!")

            # Save and export options
            if st.button("Save Search History"):
                with open("search_history.txt", "w") as file:
                    for q, a in st.session_state.history:
                        file.write(f"Question: {q}\nAnswer: {a}\n\n")
                st.write("Search history saved successfully.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
