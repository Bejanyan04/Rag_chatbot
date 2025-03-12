import streamlit as st
from langchain_openai import OpenAIEmbeddings
import os

# Import the RAG system class from your existing script
from rag_system import RAGSystem  # Ensure this matches your script filename

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Streamlit UI setup
st.title("RAG-based Question Answering System")
st.write("Enter a query below to retrieve relevant information and generate a response.")

# Load models
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
data_path = 'data.csv'
vectore_store_folder = 'faiss_docs'
mapping_file_name = "index_mapping.json"
faiss_index_file_name = "faiss_index.bin"

rag_system = RAGSystem(embedding_model, data_path, vectore_store_folder, faiss_index_file_name, mapping_file_name)

# User input
query = st.text_input("Enter your query:")
top_k = st.slider("Number of relevant documents to retrieve:", 1, 10, 3)

if st.button("Get Response") and query:
    with st.spinner("Generating response..."):
        response = rag_system.run_pipeline(query, top_k=top_k)
    st.subheader("Response:")
    st.write(response)
