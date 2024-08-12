import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def dframe2txt(uploaded_file):
    if uploaded_file.type == "text/csv":
        dframe  = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dframe = pd.read_excel(uploaded_file)
        
    if "Title" in dframe.columns and "About This Item" in dframe.columns:
        dframe = dframe[dframe['About This Item'].apply(lambda x: len(x) > 2)]
        dframe = dframe.drop_duplicates()
        data = dframe[["Title", "About This Item"]]
        path = "/home/haruki/Desktop/VS_Code/ollama/data/" + uploaded_file.name[:-4]
        data.to_csv(f"{path}.txt", sep = "\n", index = False)
        return path + ".txt"
    else:
        st.error(f"The uploaded file contains these columns: {dframe.columns}")
        return None

def txt2vectorstore(txt_file_path):
    loader = TextLoader(txt_file_path)
    docs = loader.load()

    # dividing the given data into chunks to store into vector database(CHROMADB)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Index
    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model)

    # Storing the chunks in vector store
    directory = "/home/haruki/Desktop/VS_Code/ollama/Vectorstore"
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, 
                                        collection_name="PD_BeautyProducts", persist_directory=directory)
    vectorstore.persist()

if __name__ == "__main__":
    st.subheader("Upload file to add data into VectorStore")
    uploaded_file = st.file_uploader("Upload here!", type=["csv", "xlsx"])

    if uploaded_file:
        txt_file_path = dframe2txt(uploaded_file)
        if txt_file_path:
            txt2vectorstore(txt_file_path)
            st.success("Data has been successfully added to the VectorStore.")
        else:
            st.error("Failed to process the uploaded file.")
