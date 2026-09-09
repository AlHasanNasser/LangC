from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def text_to_embedding(text: str) -> list[float]:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return embeddings.embed_query(text)


if __name__ == "__main__":
    text = input("Enter text: ")
    print(len(text_to_embedding(text)))
