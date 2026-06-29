from langchain_ollama import ChatOllama

from src.config import LLM_MODEL_NAME


def get_llm() -> ChatOllama:
    """Retourne le modele Ollama local utilise pour le raisonnement, le
    grading de pertinence et la reformulation de requete."""
    return ChatOllama(model=LLM_MODEL_NAME, temperature=0)
