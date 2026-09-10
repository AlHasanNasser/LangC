# Text Splitters and Chunking Strategies

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    Language,
    TokenTextSplitter
)
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()


SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''
def recursive_character_splitter_example():
    """Demonstrate RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"original text length: {len(SAMPLE_TEXT)}")
    print(f"RecursiveCharacterTextSplitter produced {len(chunks)} chunks:")
    print(f"chunk sizes: {[len(chunk) for chunk in chunks]}")
    print(f"first chunk preview: {chunks[0][:100]}...")





def chunk_size_comparison_example():
    """Compare different chunk sizes using RecursiveCharacterTextSplitter."""
    for size in [100, 200, 300]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Chunk size: {size}, produced {len(chunks)} chunks.")
        print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}") 


def overlap_importance_example():
    """Demonstrate the effect of chunk overlap."""

    text = "This is a sample text that will be split into chunks. The purpose is to demonstrate how chunk overlap affects the resulting chunks. Overlapping chunks can help preserve context, especially for tasks like question answering or summarization."
    #without overlap example
    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )

    #with overlap example
    splitter_with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks_with_overlap = splitter_with_overlap.split_text(text)
    chunks_no_overlap = splitter_no_overlap.split_text(text)
    
    print(f"Chunk overlap: 0, produced {len(chunks_no_overlap)} chunks.")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks_no_overlap]}")
    print(f"Chunk overlap: 20, produced {len(chunks_with_overlap)} chunks.")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks_with_overlap]}")


    print("\nChunks without overlap:")
    print(f"chunk1 end: {chunks_no_overlap[0][-20:]}...")
    print(f"chunk2 start: {chunks_no_overlap[1][:20]}...")

    print("\nChunks with overlap:")
    print(f"chunk1 end: {chunks_with_overlap[0][-20:]}...")
    print(f"chunk2 start: {chunks_with_overlap[1][:20]}... ")


def problem_solution_overlap_example():
    """Show how overlap keeps a problem connected to its solution."""

    text = """
# Problem: Users are logged out after refreshing the dashboard

Users report that the dashboard works immediately after login, but refreshing
the page sends them back to the login screen. The API returns HTTP 401 after
the refresh. If the problem and the solution land in separate chunks, a
question-answering system may return an incomplete answer.

# Investigation

The frontend stores the access token in a JavaScript variable. That variable is
cleared when the browser refreshes the page. The browser therefore makes the
next API request without an Authorization header, and the server correctly
rejects the request. The database and login endpoint are working normally.

# Solution

Store the refresh token in a secure, HttpOnly cookie and request a new access
token when the application starts. Keep the access token in memory, attach it
to API requests, and retry one failed request after refreshing the token. Do
not put refresh tokens in localStorage because JavaScript can read them there.

After deploying this change, refreshing the dashboard keeps the user signed
in. Tests should cover login, page refresh, an expired access token, and logout.
""".strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=60,
        separators=[" ", ""]
    )
    chunks = splitter.split_text(text)

    print("PROBLEM AND SOLUTION: CHUNKING WITH OVERLAP")
    print("Using chunk_size=180 and chunk_overlap=60")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ({len(chunk)} characters) ---")
        print(chunk)

    print("\nShared text between adjacent chunks:")
    for index in range(len(chunks) - 1):
        current_chunk = " ".join(chunks[index].split())
        next_chunk = " ".join(chunks[index + 1].split())
        shared_text = ""
        for size in range(min(len(current_chunk), len(next_chunk)), 0, -1):
            if current_chunk[-size:] == next_chunk[:size]:
                shared_text = current_chunk[-size:]
                break
        print(f"Chunks {index + 1} and {index + 2}: {shared_text!r}")

    
if __name__ == "__main__":
    #recursive_character_splitter_example()
    #chunk_size_comparison_example()
    overlap_importance_example()
    #problem_solution_overlap_example()