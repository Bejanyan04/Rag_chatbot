# RAG System: Retrieval-Augmented Generation Pipeline

This repository implements a **Retrieval-Augmented Generation (RAG)** system, which combines document retrieval with a large language model (LLM) to generate accurate and contextually relevant responses to user queries. The system leverages **FAISS** for efficient similarity search and **OpenAI's GPT models** for response generation.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Components](#key-components)
   - [Embedding Model](#embedding-model)
   - [FAISS Index](#faiss-index)
   - [Document Store](#document-store)
   - [LLM for Response Generation](#llm-for-response-generation)
3. [How It Works](#how-it-works)
   - [Step 1: Embedding Generation](#step-1-embedding-generation)
   - [Step 2: FAISS Index Creation](#step-2-faiss-index-creation)
   - [Step 3: Document Retrieval](#step-3-document-retrieval)
   - [Step 4: Response Generation](#step-4-response-generation)
4. [Usage](#usage)
   - [Initialization](#initialization)
   - [Running the Pipeline](#running-the-pipeline)
5. [Example](#example)
6. [Future Enhancements](#future-enhancements)

---

## Overview

The RAG system is designed to answer user queries by retrieving relevant documents from a dataset and generating responses using a large language model. It consists of the following key steps:
1. **Embedding Generation**: Convert text data into vector embeddings.
2. **FAISS Index Creation**: Build an efficient index for similarity search.
3. **Document Retrieval**: Retrieve the most relevant documents for a given query.
4. **Response Generation**: Use an LLM to generate a response based on the retrieved documents.

---

## Key Components

### Embedding Model
- **Purpose**: Converts text data (e.g., titles and descriptions) into vector embeddings.
- **Implementation**: Uses a pre-trained embedding model (e.g., OpenAI's `text-embedding-ada-002`).
- **Output**: A list of vector embeddings for each document in the dataset.

### FAISS Index
- **Purpose**: Enables fast and efficient similarity search over vector embeddings.
- **Implementation**: Uses FAISS (Facebook AI Similarity Search) to create an index of embeddings.
- **Output**: A FAISS index that maps document IDs to their corresponding embeddings.

### Document Store
- **Purpose**: Stores metadata (e.g., titles, text) for retrieved documents.
- **Implementation**: Uses an in-memory document store (`InMemoryDocstore`) to map FAISS index IDs to document IDs.
- **Output**: A mapping between FAISS index IDs and document store IDs.

### LLM for Response Generation
- **Purpose**: Generates a response to the user query based on the retrieved documents.
- **Implementation**: Uses OpenAI's GPT model (e.g., `gpt-3.5-turbo`) to generate responses.
- **Output**: A natural language response to the user query.

---

## How It Works

### Step 1: Embedding Generation
- The system combines the "Title" and "Text" columns of the dataset into a single text string.
- It uses the embedding model to convert these text strings into vector embeddings.
- The embeddings are stored in the dataset for later use.

### Step 2: FAISS Index Creation
- The system creates a FAISS index using the generated embeddings.
- The index is saved to disk for future use, enabling fast retrieval without recomputing embeddings.

### Step 3: Document Retrieval
- When a user query is received, the system converts the query into a vector embedding.
- It uses the FAISS index to retrieve the top-k most relevant documents based on similarity.
- The retrieved documents are fetched from the document store using the index-to-docstore mapping.

### Step 4: Response Generation
- The system passes the retrieved documents and the user query to the LLM.
- The LLM generates a response based on the context provided by the retrieved documents.

---

## Usage From notebook

### Initialization
To initialize the RAG system, provide the following:
- An embedding model (e.g., OpenAI's embedding model).
- The path to the dataset (CSV file with 'ID'  "Title" , "Text", 'Vector' columns).
- The folder path to store or load the FAISS index and mappings.
- The file names for the FAISS index and index-to-docstore mapping.

```python

rag_system = RAGSystem(
    embedding_model=embedding_model,
    data_path="data.csv",  # Path to your dataset
    vectore_store_folder="faiss_docs",  # Folder to store FAISS index and mappings
    index_file_name="index.faiss",  # File name for the FAISS index
    docstore_to_id_file_name="index_mappings.json",  # File name for the index-to-docstore mapping
    llm_model="gpt-3.5-turbo"  # OpenAI model for response generation
)
rag_system.run_pipeline(query="What is quantum supremacy?", save_emb=False, top_k=3)


---

## Usage of streamlit app
To run streamlit app use following command

```python
stramlit run streamlit_app.py