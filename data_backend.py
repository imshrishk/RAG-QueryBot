import streamlit as st
import pandas as pd
import chardet
import shelve
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import os

@st.cache_data
def load_and_prepare_data(excel_path, text_path, chunk_size, chunk_overlap, cache_path='data_cache.db'):
    with shelve.open(cache_path) as cache:
        if 'chunks' in cache:
            return cache['chunks']
        
        # Load Excel data in chunks
        data = pd.read_excel(excel_path, chunksize=5000)
        combined_data = pd.concat(data)
        combined_data = combined_data[["url", "title", "about_this_item"]]
        combined_data.to_csv(text_path, sep="\n", index=False)

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

@st.cache_resource
def create_vector_store(_chunks):
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Chroma vector store and add documents
    vectorstore = Chroma.from_documents(documents=_chunks, collection_name="rag_chroma", embedding=embeddings)
    return vectorstore

def main():
    st.title("Data Backend")

    # Sidebar for user adjustments
    st.sidebar.header("Settings")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=1000, value=400, step=50)

    # File upload for Excel data
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            excel_path = tmp_file.name
        
        text_path = "data.txt"

        try:
            # Load and prepare data
            chunks = load_and_prepare_data(excel_path, text_path, chunk_size, chunk_overlap)
            vectorstore = create_vector_store(chunks)

            # Display vector store info
            st.write("Vector store created and documents added successfully.")

            # Show raw data and chunks for debugging
            dframe = pd.read_excel(excel_path)
            with st.expander("Show Raw Data"):
                st.write(dframe)
            with st.expander("Show Document Chunks"):
                st.write(chunks)

            # Save data to session state for passing to user_query.py
            st.session_state.vectorstore = vectorstore
            st.session_state.data = dframe

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            os.remove(excel_path)

if __name__ == "__main__":
    main()
