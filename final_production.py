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

]

class HybridSearch:
    def __init__(
        self,
        documents: list[Document],
        bm25_weight: float = 0.5,
        k: int = 4,
        collection_name: str = "hybrid_search_collection",
    ):
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be between 0.0 and 1.0")
        if k < 1:
            raise ValueError("k must be at least 1")

        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = 1.0 - bm25_weight
        self.k = k
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vector_store = Chroma.from_documents(
            documents,
            self.embeddings_model,
            collection_name=collection_name,
        )
        self.vector_retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=self.k)
        self._build_hybrid_retriever()

    def _build_hybrid_retriever(self):
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight],
            k=self.k,
        )


    def search(self, query:str) -> list[Document]:
        print(f"\nHybrid search results for query: '{query}'")
        results = self.hybrid_retriever.invoke(query)
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Metadata: {doc.metadata})")
        return results

    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        self.vector_store.add_documents(documents)

        self.documents.extend(documents)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents, k=self.k)
        self._build_hybrid_retriever()


if __name__ == "__main__":
    retriever = HybridSearch(documents=documents, bm25_weight=0.5, k=4)
    retriever.search("How do I fix internet connectivity issues?")
        