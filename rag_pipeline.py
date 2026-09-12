"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""


import os
from pydoc import doc

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.chat_models import init_chat_model

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm_google = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""



def create_kb():
    """Create a vector store from knowledge base."""

    # split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    chunks = splitter.split_documents([doc])

    # create a vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store




def demo_basic_rag():
    """Demonstrate a basic RAG pipeline."""
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})


    # define a prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    llm_google = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)


    def format_docs(docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join([ doc.page_content for doc in docs])  

    
    # define a simple RAG chain
    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever | format_docs }
        | prompt_template
        | llm_google
        | StrOutputParser()
    )

    # run the RAG chain
    question = "What is LangChain?"
    answer = rag_chain.invoke(question)
    print(answer)



def demo_rag_with_sources():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):"""
    )


    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
            
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_google
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")



if __name__ == "__main__":
     # demo_basic_rag()
     demo_rag_with_sources()
    # demo_rag_with_fallback()
    # demo_structured_rag()
    #exercise_document_qa()







