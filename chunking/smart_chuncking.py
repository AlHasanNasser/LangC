import os
from typing import Literal

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

ChunkingMode = Literal["auto", "semantic", "recursive"]


def _build_semantic_splitter(chunk_size: int = 500, chunk_overlap: int = 50):
    """Create a semantic splitter using the configured Google GenAI embeddings."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Google GenAI API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY to use semantic chunking."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.8,
        buffer_size=1,
        min_chunk_size=max(80, chunk_size // 3),
    )


def _build_recursive_splitter(chunk_size: int = 500, chunk_overlap: int = 50, separators=None):
    """Create a recursive splitter as the fallback strategy."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )


def smart_chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunking_type: ChunkingMode = "auto",
    separators=None,
):
    """
    Smart chunking helper.

    Default behavior is automatic: semantic chunking is tried first, and recursive
    chunking is used as a safe fallback if semantic chunking fails (for example when
    the text is too large, the embedding model is unavailable, or the API key is missing).

    If the caller explicitly sets chunking_type to 'recursive', this always forces
    recursive chunking. If set to 'semantic', it forces semantic chunking and does not
    fall back automatically.
    """
    if not text or not text.strip():
        return []

    normalized_mode = (chunking_type or "auto").lower()

    if normalized_mode == "recursive":
        return _build_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        ).split_text(text)

    if normalized_mode == "semantic":
        splitter = _build_semantic_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    # Auto mode: prefer semantic chunking, fallback to recursive on failure
    try:
        splitter = _build_semantic_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    except Exception:
        return _build_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        ).split_text(text)


def main():
    sample_text = """
    # Authentication Guide

    ## OAuth2 Authentication
    To authenticate with our API, you need OAuth2 credentials.
    First, obtain a client_id and client_secret from the developer portal.
    Make a POST request to /oauth/token with grant_type=client_credentials.
    The response contains an access_token valid for 3600 seconds.

    ## Rate Limiting
    Our API implements rate limiting using a token bucket algorithm.
    Free tier: 100 requests per minute.
    Pro tier: 1000 requests per minute.
    Enterprise tier: Custom limits.

    ## Error Handling
    All errors return a standard JSON format.
    The code field contains a machine-readable error code.
    The message field contains a human-readable description.
    """.strip()

    print("== Auto mode ==")
    auto_chunks = smart_chunk_text(sample_text, chunk_size=180, chunk_overlap=20)
    print(f"Auto produced {len(auto_chunks)} chunks")

    print("\n== Forced recursive mode ==")
    recursive_chunks = smart_chunk_text(
        sample_text,
        chunk_size=180,
        chunk_overlap=20,
        chunking_type="recursive",
    )
    print(f"Recursive produced {len(recursive_chunks)} chunks")

    print("\n== First chunk preview ==")
    for idx, chunk in enumerate(auto_chunks[:2], start=1):
        print(f"Chunk {idx}: {chunk[:180]}...")


if __name__ == "__main__":
    main()
