import os
import json
import faiss
import numpy as np
import pandas as pd
import argparse
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import json
import re
import pandas as pd
from dotenv import load_dotenv
import os

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY')

class RAGSystem:
    def __init__(self, embedding_model, data_path, vectore_store_folder, index_file_name, docstore_to_id_file_name, llm_model="gpt-3.5-turbo"):
        """
        Initializes the RAG system with an embedding model and an LLM.

        Args:
            embedding_model: The embedding function to convert text into vectors.
            llm_model (str): The OpenAI model to use for generating responses.
        """
        self.embedding_model = embedding_model
        self.llm = ChatOpenAI(model=llm_model)
        self.data_path =  data_path
        self.data = None
        self.vector_store = None
        self.index = None
        self.index_to_docstore_id  = None
        self.vectore_store_folder = vectore_store_folder
        os.makedirs(self.vectore_store_folder, exist_ok=True)
        self.mapping_full_path= os.path.join(self.vectore_store_folder, docstore_to_id_file_name)
        self.index_full_path = os.path.join(self.vectore_store_folder, index_file_name)
      
        self.initialize_docstore()


    
    def create_embeddings(self,save_emb = False) -> tuple:
        """
        Generates vector embeddings for a given dataset using the specified embedding model.
        It combines the 'Title' and 'Text' columns before generating embeddings.

        Args:
            data_path (str): The file path to the CSV dataset containing 'Title' and 'Text' columns.
            embedding_model: An embedding model instance with a method `embed_query` that converts text into a vector.
        """
        
        data = pd.read_csv(self.data_path)

        # Combine "Title" and "Text" columns
        text_data = (data["Title"] + " " + data["Text"]).tolist()
        # Generate embeddings for combined text
        vector_list = np.array([self.embedding_model.embed_query(text) for text in text_data])
        # Store vectors in the DataFrame
        data['Vectors'] = list(vector_list)
        self.data = data
        if save_emb:
            data.to_csv("data_with_embeddings.csv", index=False)
        return data, vector_list
    

    def build_faiss_index(self, vector_emb: np.ndarray) -> faiss.IndexFlatL2:
        """
        Creates a FAISS index and adds vector embeddings to it.

        Args:
            vector_emb (np.ndarray): NumPy array of vector embeddings. 
        """
        embedding_dim = vector_emb.shape[1]  # Get the embedding dimension
        index = faiss.IndexFlatL2(embedding_dim)  # Create a FAISS index using L2 distance
        index.add(vector_emb)  # Add vectors to the index
        return index
    
    def get_index_to_docstore_id_mapping(self):
        """
        Retrieves or creates a mapping between FAISS index IDs and document store IDs.

        Returns:
            dict: A dictionary mapping FAISS index IDs to document store IDs.
        """
        if os.path.exists(self.mapping_full_path):
            index_to_docstore_id = self.load_index_to_docstore_mapping(self.mapping_full_path)
        else:
            index_to_docstore_id = {i: str(self.data.loc[i, "ID"]) for i in range(len(self.data))}  # Mapping
            self.index_to_docstore_id = index_to_docstore_id
            self.save_index_to_docstore_mapping(self.index_to_docstore_id, self.mapping_full_path)
        return index_to_docstore_id
    
    def get_faiss_index(self):
        """
        Retrieves or creates a FAISS index.

        Returns:
            faiss.IndexFlatL2: The FAISS index.
        """
        if os.path.exists(self.index_full_path):
            index = self.load_faiss_index(self.index_full_path)
        else:   
            data, embeddings = self.create_embeddings()
            index = self.build_faiss_index(embeddings)
            self.save_faiss_index(index, self.index_full_path)

        return index

    def save_faiss_index(self,index: faiss.IndexFlatL2, file_path):
        """
        Saves the FAISS index to disk.
        Args:
            index (faiss.IndexFlatL2): The FAISS index to be saved.
            file_path (str): The file path where the index will be stored.
        """
        faiss.write_index(index, file_path)
        print(f"FAISS index saved to {file_path}")
        

    def load_faiss_index(self, file_path: str) -> faiss.IndexFlatL2:
        """
        Loads a FAISS index from disk.

        Args:
            file_path (str): The file path of the saved FAISS index.

        Returns:
            faiss.IndexFlatL2: The loaded FAISS index.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No FAISS index found at {file_path}")
        
        index = faiss.read_index(file_path)
        print(f"FAISS index loaded from {file_path}")
        return index

    def save_index_to_docstore_mapping(self,index_to_docstore_id, file_path):
        """
        Save the index to document mapping to a JSON file.

        Args:
        - index_to_docstore_id (dict): A dictionary mapping index to document store ID.
        - file_path (str): The path where the mapping will be saved.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(index_to_docstore_id, f)
            print(f"Mapping successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving mapping: {e}")


    def load_index_to_docstore_mapping(self, file_path):
        """
        Load the index to document mapping from a JSON file.

        Args:
        - file_path (str): The path from which the mapping will be loaded.

        Returns:
        - dict: A dictionary mapping index to document store ID.
        """
        try:
            with open(file_path, 'r') as f:
                index_to_docstore_id = json.load(f)
            print(f"Mapping successfully loaded from {file_path}")
            return index_to_docstore_id
        except Exception as e:
            print(f"Error loading mapping: {e}")
            return None

    def create_faiss_vector_store(self,index: faiss.IndexFlatL2, df, embedding_function, index_to_docstore_id) -> FAISS:
        """
        Creates a FAISS vector store by mapping document IDs to the FAISS index.

        Args:
            index (faiss.IndexFlatL2): The FAISS index with stored embeddings.
            df (pd.DataFrame): The DataFrame containing document data.
            embedding_function: The embedding function used for vectorization.

        """
        docstore = InMemoryDocstore()  # In-memory document 
        
        vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        return vector_store
    

    def retrieve_documents(self, query, top_k=3):
        """
        Converts a query into an embedding and retrieves the top_k most relevant documents.

        Args:
            query (str): The user query.
            top_k (int): Number of documents to retrieve.

        Returns:
            List of relevant document texts.
        """
        if self.index is None or self.index_to_docstore_id is None:
            raise ValueError("FAISS index or mapping not initialized.")

        query_vector = np.array(self.embedding_model.embed_query(query))  # Convert query to vector
        distances, indices = self.index.search(np.array([query_vector]), top_k)

        retrieved_docs = []
        for idx in indices[0]:  
            doc_id = self.index_to_docstore_id[str(idx)]
            doc = self.data[self.data["ID"] == int(doc_id)].iloc[0]
            retrieved_docs.append(f"Title: {doc['Title']}\nText: {doc['Text']}")

        return retrieved_docs
    
    def generate_response(self, query, top_k=3):
        """
        Retrieves relevant documents and generates a response using the LLM.

        Args:
            query (str): The user query.
            top_k (int): Number of documents to retrieve.

        Returns:
            str: The generated response.
        """
        retrieved_docs = self.retrieve_documents(query, top_k)
        context = "\n\n".join(retrieved_docs)
    # Define the prompt template
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert assistant providing detailed and accurate responses based on provided context.
            The answers must be concrete and strictly refer to the question. Don't add anything outside the 
            knowledge base.

            Here is some relevant information:

            {context}

            Now, answer the following question:

            {query}
            """
        )

        # Create the chain
        chain = prompt_template | self.llm | StrOutputParser()

        # Invoke the chain with the input
        response = chain.invoke({"context": context, "query": query})
        return response
    
    def initialize_docstore(self, save_emb=False):
        if not self.vector_store or  not self.index_to_docstore_id or not self.index:
            if not self.vector_store:
                _, vector_list = self.create_embeddings(save_emb=save_emb)
                self.vectore_store = self.create_faiss_vector_store(self.index, self.data, self.embedding_model, self.index_to_docstore_id)
            if not self.index:
                self.index = self.get_faiss_index()
            if not self.index_to_docstore_id:
                self.index_to_docstore_id  = self.get_index_to_docstore_id_mapping()


    def run_pipeline(self,query, save_emb=False, top_k=3):
        """
        Executes the full RAG pipeline, including embedding generation, FAISS index creation, 
        document retrieval, and response generation.

        This method checks if the FAISS index, vector store, and index-to-docstore mapping 
        are already initialized. If not, it generates embeddings, builds the FAISS index, 
        and creates the necessary mappings. Finally, it retrieves relevant documents and 
        generates a response to the user's query.

        Args:
            query (str): The user query for which a response is to be generated.
            save_emb (bool, optional): Whether to save the generated embeddings to a CSV file. 
                                    Defaults to False.
            top_k (int, optional): The number of relevant documents to retrieve for the query. 
                                Defaults to 3.

        Returns:
            None: The generated response is printed to the console.
        
        """
        if not self.vector_store or not self.index_to_docstore_id or not self.index:
            self.initialize_docstore()


        else:
            print('Faiss information is already loaded')
        print(self.generate_response(query, top_k=top_k))



def main():
    parser = argparse.ArgumentParser(description="Run RAG system with a query")
    parser.add_argument("query", type=str, help="Query to process")
    args = parser.parse_args()
    
    data_path = 'data.csv'
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    llm_model = "gpt-3.5-turbo"
    vectore_store_folder = 'faiss_docs'
    mapping_file_name = "index_mapping.json"
    faiss_index_file_name =  "faiss_index.bin"

    rag_system = RAGSystem(embedding_model, data_path, 
                        vectore_store_folder, faiss_index_file_name, 
                        mapping_file_name, llm_model)

        
    # Process the query
    rag_system.run_pipeline(query=args.query)
    
    
if __name__ == "__main__":
    main()
