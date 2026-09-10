from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
load_dotenv()



embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def basic_embedding_example():
    text = "LangChain is a framework for developing applications powered by language models."
    embedding_vector = embeddings_model.embed_query(text)
    print(f"Query: {text}")
    print(f"Vecter dimension: {len(embedding_vector)}")
    print(f"Embedding vector preview: {embedding_vector[:10]}...")  # Print first 10 dimensions for brevity
    print(f"Vector norm: {np.linalg.norm(embedding_vector):.4f}")
    
def batch_embedding_example():
    texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "Vector stores are databases optimized for storing and searching embeddings."
    ]
    embedding_vectors = embeddings_model.embed_documents(texts)

    for i, vector in enumerate(embedding_vectors):
        print(f"\nText {i+1}: {texts[i]}")
        print(f"Vector dimension: {len(vector)}")
        print(f"Embedding vector preview: {vector[:10]}...")  # Print first 10 dimensions for brevity
        print(f"Vector norm: {np.linalg.norm(vector):.4f}")

def similarity_search_example():
    # Sample documents
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "Vector stores are databases optimized for storing and searching embeddings.",
        
    ]
    
    # Create embeddings for the documents
    document_embeddings = embeddings_model.embed_documents(documents)
    
    # Query text
    query = "What is LangChain?"
    query_embedding = embeddings_model.embed_query(query)
    
    # Compute cosine similarity between the query and each document
    similarities = [np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)) for doc_emb in document_embeddings]
    
    

    ranked_docs= sorted(zip(similarities, documents), key=lambda x: x[0], reverse=True)

    
    print(f"Query: {query}")
    print("Ranked documents by similarity:")
    for score, doc in ranked_docs:
        print(f"Document: {doc}")
        print(f"Similarity score: {score:.4f}\n")
    

if __name__ == "__main__":
    #basic_embedding_example()
    #batch_embedding_example()
    similarity_search_example()
