import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


SAMPLE_TEXT = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
""".strip()


class LocalDeterministicEmbeddings:
    """Simple embedding fallback for offline demos."""

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        cleaned = text.lower()
        total = sum(ord(char) for char in cleaned)
        return [(total + i * 17) % 11 / 10.0 for i in range(3)]


def get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("No Google API key found. Using local deterministic embeddings for offline demo.")
    return LocalDeterministicEmbeddings()


def recursive_chunking_demo():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(SAMPLE_TEXT)
    
    return chunks


def semantic_chunking_demo():
    embeddings = get_embeddings()
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.8,
        buffer_size=1,
        min_chunk_size=80,
    )
    chunks = splitter.split_text(SAMPLE_TEXT)
    
    return chunks


def compare_chunking_strategies():
    recursive_chunks = recursive_chunking_demo()
    semantic_chunks = semantic_chunking_demo()

    print("\n=== Difference Summary ===")
    print("Recursive chunking is based on character boundaries and separators.")
    print("Semantic chunking groups text by meaning and topic continuity.")
    print(f"Recursive chunks: {len(recursive_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")
    print("\nIn practice:")
    print("- Recursive splitting is faster and better for general text structure.")
    print("- Semantic splitting usually keeps related ideas together, which is stronger for RAG retrieval.")


recursive_vector_store = Chroma.from_texts(
    texts=recursive_chunking_demo(),
    embedding=get_embeddings(),
    collection_name="recursive_chunks",
)
semantic_vector_store = Chroma.from_texts(
    texts=semantic_chunking_demo(),
    embedding=get_embeddings(),
    collection_name="semantic_chunks",
)

test_query = "How do I authenticate with the API?"

def queries(test_query, vector_store,name):
    print(f"\n=== Querying {name} Vector Store ===")
    print(f"Query: {test_query}")
    results = vector_store.similarity_search(test_query, k=2)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Text: {result.page_content}")
        print(f"Metadata: {result.metadata}")




if __name__ == "__main__":
    #compare_chunking_strategies()
    queries(test_query, recursive_vector_store, "Recursive")
    queries(test_query, semantic_vector_store, "Semantic")
