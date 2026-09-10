

from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv


load_dotenv()


embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")



# Documents with both semantic content and specific identifiers
documents = [
    Document(page_content="Product SKU12345 is a high-quality widget designed for efficiency.", metadata={"type": "product"}),
    Document(page_content="For internet connectivity issues, first check your router and cable connections.", metadata={"type": "troubleshooting"}),
    Document(page_content="Error code E_CONN_REFUSED indicates that the server refused the connection request.", metadata={"type": "error_code"}),
    Document(page_content="Authentication failed due to invalid credentials. Please verify your username and password . use OAuth2 for secure authentication.", metadata={"type": "authentication"}),
    Document(page_content="Router configuration guide: access the admin panel at 192.168.1.1 to modify settings.", metadata={"type": "configuration"}),
    Document(page_content="WCAG 2.1 compliance checklist: ensure your website meets accessibility standards.", metadata={"type": "compliance"}),
]

print(f"Loaded {len(documents)} documents with metadata.")

vector_store = Chroma.from_documents(documents, embeddings_model, collection_name="hybrid_search_collection")

vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2}) 
print("vector_retriever created successfully.")

# Notice in bm25_retriver we don't pass the embedding model because BM25 is a traditional term-based retrieval method that doesn't require embeddings.
# It relies on the frequency and distribution of terms in the documents to rank them based on their relevance to the query.

bm25_retriever = BM25Retriever.from_documents(documents, k=2)
print("bm25_retriever created successfully.")

#here is hybrid retriever that combines both vector and BM25 retrieval methods. It uses the EnsembleRetriever to combine the results from both retrievers, allowing for a more comprehensive search that leverages both semantic understanding (from the vector retriever) and keyword matching (from the BM25 retriever).
hybrid_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])
print("hybrid_retriever created successfully.")


def test_query(query,name,retriever):
    print(f"\n{name} results for query: '{query}'")
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content} (Metadata: {doc.metadata})")
    return results

test_queries = [
    "How do I fix internet connectivity issues?",
    " E_CONN_REFUSED?",
    "How do I authenticate?",
    "router configuration",
    " WCAG 2.1 compliance ",
]

for query in test_queries:
    test_query(query, "Vector Retriever", vector_retriever)
    test_query(query, "BM25 Retriever", bm25_retriever)
    test_query(query, "Hybrid Retriever", hybrid_retriever)