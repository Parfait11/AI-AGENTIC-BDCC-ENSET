# PARTIE 1 : LLM local avec outils (Tools)

from langchain.tools import tool
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()  # si openai, groq ...

model = ChatOllama(
    model="llama3.2:3b",  # ou mistral, gemma, etc.
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """multiply two integers."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """divide two integers."""
    return a / b


tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}

model_with_tools = model.bind_tools(tools)
