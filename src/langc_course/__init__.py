from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")


from langchain_google_genai import ChatGoogleGenerativeAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main() -> None:
    llm_google = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)
    response = llm_google.invoke("say 'setup complete ' in one word ")
    print(response.content)

 

if __name__ == "__main__":
    main() 
    