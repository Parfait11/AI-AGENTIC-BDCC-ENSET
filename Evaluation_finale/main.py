"""Assistant Agentic RAG specialise dans la gestion des epidemies de cholera.

Lancer avec : uv run python main.py
"""
from langchain.messages import HumanMessage

from src.config import CHROMA_DIR
from src.graph import graph


def main() -> None:
    if not CHROMA_DIR.exists():
        print("Vectorstore introuvable. Lancez d'abord : uv run python ingest.py")
        return

    config = {"configurable": {"thread_id": "session-cli"}}
    print(" Assistant Cholera - Surveillance et Reponse Epidemique ")
    print("Posez vos questions sur la surveillance, la reponse de terrain,")
    print("la rehydratation orale (SRO) et la prevention du cholera.")
    print("Tapez 'exit' pour quitter.\n")

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        print(f"Assistant : {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()