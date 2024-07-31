import streamlit as st
import pandas as pd
import chardet
import shelve
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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

def create_vector_store(chunks):
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Chroma vector store and add documents
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="rag_chroma", embedding=embeddings)
    return vectorstore

def main():
    st.title("Advanced Shopping Assistant")

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
                - If the answer to the question can be determined from the context, response with the names of all the different products matching the criteria(the names MUST be different), separated by commas.
                - You have to also write the category(Only one) and sub-category it belongs to as the heading of the answer. Categories are [Makeup, Skin Care, Hair Care, Fragrance, Tools & Accessories, Shave & Hair Removal, Personal Care, Salon & Spa Equipment]  
                - You must thoroughly check the query for number of answer required and provide the specific number of answer. If none is provided, provide 3 most relevant answers.
                - After picking the category, you MUST choose a sub-category from the following for each specific category that you found from the query
                    Makeup: [Body, Eyes, Face, Lips, Makeup Palettes, Makeup Remover, Makeup Sets]
                    Skin Care: [Body, Eyes, Face, Lip Care, Maternity, Sets & Kits, Sunscreens & Tanning Products]
                    Hair Care: [Detanglers, Hair Accessories, Hair Coloring Products, Hair Cutting Tools, (Hair Extensions, Wigs & Accessories), Hair Fragrances, Hair Loss Products, Hair Masks, (Hair Perms, Relaxers & Texturizers), Hair Treatment Oils, Scalp Treatments, Shampoo & Conditioner, Styling Products]
                    Fragrance: [Children's, Dusting Powders, Men's, Sets, Women's]
                    Foot, Hand & Nail Care: []
                    Tools & Accessories: [Bags & Cases, Bathing Accessories, Cotton Balls & Swabs, Makeup Brushes & Tools, Mirrors, Refillable Containers, Shave & Hair Removal, Skin Care Tools]
                    Shave & Hair Removal: [Men's, Women's]
                    Personal Care: [Bath & Bathing Accessories, Deodorants & Antiperspirants, Lip Care, Oral Care, Piercing & Tattoo Supplies, Scrubs & Body Treatments, Shave & Hair Removal]
                    Salon & Spa Equipment: [Hair Drying Hoods, Galvanic Facial Machines, Handheld Mirrors, High-Frequency Facial Machines, Manicure Tables, Professional Massage Equipment, Salon & Spa Stools, Spa Beds & Tables, Salon & Spa Chairs, Spa Storage Systems, Spa Hot Towel Warmers]

                Question: {question}  
                =====================================================================
                Context: {context}   
                =====================================================================
                """

                prompt = ChatPromptTemplate.from_template(template)

                # Local LLM
                ollama_llm = "gemma"
                model_local = ChatOllama(model=ollama_llm, temperature=0)

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
        finally:
            os.remove(excel_path)

if __name__ == "__main__":
    main()
