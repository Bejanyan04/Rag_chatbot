import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from rag_system import RAGSystem

# Initialize the RAG system for testing
@pytest.fixture
def rag_system():
    from langchain_openai import OpenAIEmbeddings

    data_path = 'data.csv'
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    llm_model = "gpt-3.5-turbo"
    vectore_store_folder = 'faiss_docs'
    mapping_file_name = "index_mapping.json"
    faiss_index_file_name =  "faiss_index.bin"

    rag_system = RAGSystem(embedding_model, data_path, 
                        vectore_store_folder, faiss_index_file_name, 
                        mapping_file_name, llm_model)

    return rag_system

# Test embedding generation
def test_embedding_generation(rag_system):
    data, vector_list = rag_system.create_embeddings(save_emb=False)
    assert len(vector_list) > 0, "Embeddings were not generated."
    assert len(data) == len(vector_list), "Mismatch between data rows and embeddings."

# Test FAISS index creation
def test_faiss_index_creation(rag_system):
    _, vector_list = rag_system.create_embeddings(save_emb=False)
    index = rag_system.build_faiss_index(vector_list)
    assert index.ntotal == len(vector_list), "FAISS index was not created correctly."


# Test document retrieval
def test_document_retrieval(rag_system):
    query = "What is quantum supremacy?"
    retrieved_docs = rag_system.retrieve_documents(query, top_k=3)
    assert len(retrieved_docs) == 3, "Incorrect number of documents retrieved."
    for doc in retrieved_docs:
        assert "Title" in doc and "Text" in doc, "Retrieved document format is incorrect."

# Test response generation
def test_response_generation(rag_system):
    query = "What is quantum supremacy?"
    response = rag_system.generate_response(query, top_k=3)
    assert len(response) > 0, "No response was generated."
    assert "quantum" in response.lower(), "Response does not contain expected keywords."


# DeepEval test for answer relevancy
def test_answer_relevancy(rag_system):
    query = "What is quantum supremacy?"
    response = rag_system.generate_response(query, top_k=3)

    # Create a DeepEval test case
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        context=["Quantum supremacy is the milestone achieved by Google's quantum computer, Sycamore."]
    )

    # Define the metric (Answer Relevancy)
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4-0125-preview",
        include_reason=True
    )
    # Assert the test case
    assert_test(test_case, [metric])
