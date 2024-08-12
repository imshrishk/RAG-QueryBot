## _Retrieval Augmented Generation(RAG) using Ollama, Chromadb and Langchain_

## Flow chart of RAG pipeline
![structure](https://github.com/user-attachments/assets/5ff61be2-f85d-4f6b-9c40-1e4e20389feb)

*Retrieval Augmented Generation(RAG)*<br />
It is a natural language processing (NLP) technique that combines the strengths of both retrieval- and generative-based 
artificial intelligence (AI) models. RAG AI can deliver accurate results that make the most of pre-existing knowledge but 
can also process and consolidate that knowledge to create unique, context-aware answers, instructions, or explanations in 
human-like language rather than just summarizing the retrieved data. <br />

A typical RAG application comprises two main components: 
- Indexing
- Retrieval and Generation

Initially, data is extracted from private sources and partitioned to accommodate long text documents while preserving their semantic relations.
Subsequently, this partitioned data is stored in a vector database, such as ChromaDB or Pinecone. In our case, we utilize ChromaDB for indexing purposes.

Next, in the Retrieval and Generation phase, relevant data segments are retrieved from storage using a Retriever. These segments, 
along with the user query, are then incorporated into the model prompt. Our approach employs an open-source local LLM, Gemma 7b, with the assistance of Ollama.

## Objective 
To make the model(chat-bot) act like a shopping assistant for us 

## Should Know 
- LLM - Large Language Model
- Langchain - a framework designed to simplify the creation of applications using LLMs
- Vector database - a database that organizes data through high-dimmensional vectors
- ChromaDB - vector database
- RAG - Retrieval Augmented Generation (see below more details about RAGs)
- Ollama - a tool that allows you to run open-source large language models (LLMs) locally on your machine

## Prerequisites
- python(3.x)
- pip(package manager)

## Installation
- *Clone the repo*
- *Install all the dependencies*
- *Run the command-* "streamlit run PD_RAG.py"
