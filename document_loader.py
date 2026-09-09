import os 
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from dotenv import load_dotenv

load_dotenv()

"""
def load_document():
    # Create a temporary text file so this example can run without an input file.
    with tempfile.TemporaryDirectory() as temporary_directory:
        document_path = Path(temporary_directory) / "course_notes.txt"
        document_path.write_text(
            "This document was loaded with LangChain.",
            encoding="utf-8",
        )

        # TextLoader returns a list of LangChain Document objects.
        documents = TextLoader(str(document_path), encoding="utf-8").load()

        # Add application-specific metadata to every loaded document.
        for document in documents:
            document.metadata.update(
                {
                    "document_type": "text",
                    "description": "Course notes loaded with LangChain",
                }
            )
        return documents"""



def load_pdf_document(pdf_path="docs/sample-local-pdf.pdf"):
    # PyPDFLoader creates one Document object for each page in the PDF.
    documents = PyPDFLoader(pdf_path).load()

    # Keep the loader metadata and add metadata useful to this application.
    for document in documents:
        document.metadata.update(
            {
                "document_type": "pdf",
                "description": "PDF document loaded with LangChain",
            }
        )
    return documents


if __name__ == "__main__":
    # Run the text example first.
 
    # Put your PDF at docs/document.pdf to run the PDF example.
    pdf_path = Path("docs/sample-local-pdf.pdf")
    if pdf_path.exists():
        pdf_documents = load_pdf_document(str(pdf_path))
        print(f"Loaded {len(pdf_documents)} PDF page(s).")
        print(pdf_documents[0].metadata)
        print(pdf_documents[0].page_content)
    else:
        print(f"PDF not found: {pdf_path}")