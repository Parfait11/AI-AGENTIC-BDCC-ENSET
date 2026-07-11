"""
TP LangChain - Agent "Chef personnel"

Fonctionnalites couvertes:
- system message
- memoire conversationnelle avec InMemorySaver
- outil de recherche web (Tavily)
- proposition de plats selon les ingredients et preferences utilisateur
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient


load_dotenv()


def build_web_search_tool():
    """Construit un tool Tavily avec message d'erreur propre si la cle API manque."""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        @tool("web_search")
        def web_search(query: str) -> Dict[str, Any]:
            """
            Recherche web indisponible tant que TAVILY_API_KEY n'est pas configuree.

            Args:
                query: requete de recherche culinaire
            """
            return {
                "status": "error",
                "message": (
                    "La recherche web est indisponible. "
                    "Ajoutez TAVILY_API_KEY dans le fichier .env."
                ),
                "query": query,
            }

        return web_search

    tavily_client = TavilyClient(api_key=api_key)

    @tool("web_search")
    def web_search(query: str) -> Dict[str, Any]:
        """
        Recherche des informations culinaires sur le web.

        Args:
            query: question sur une recette, technique ou ingredient
        """
        return tavily_client.search(query, max_results=3)

    return web_search


def build_agent():
    """Initialise le modele, le tool web et la memoire."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

    model = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    system_prompt = """
Tu es un chef cuisinier personnel intelligent.

Ton role:
- aider l'utilisateur a cuisiner avec les ingredients disponibles
- memoriser ses preferences, allergies, regime alimentaire et contraintes
- utiliser l'outil web_search uniquement si une information culinaire recente
  ou specifique est necessaire
- proposer des plats realistes, simples et adaptes a la situation

Regles de reponse:
- reponds toujours en francais
- tiens compte de l'historique de conversation
- si des informations manquent, fais une hypothese raisonnable et dis-la
- privilegie les recettes faisables avec peu d'ingredients et en moins de 20 minutes
- n'invente pas des ingredients disponibles si l'utilisateur ne les a pas mentionnes
- si tu proposes un ingredient non disponible, marque-le clairement comme optionnel
- reste concis: pas d'introduction longue, pas de formule repetitive
- n'utilise web_search que si c'est vraiment utile
- structure la reponse avec:
  1. Plat propose
  2. Pourquoi il convient
  3. Ingredients utilises
  4. Etapes courtes
  5. Variante ou conseil
"""

    agent = create_agent(
        model=model,
        tools=[build_web_search_tool()],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent


def ask_agent(agent, user_text: str, thread_id: str = "chef-thread-1") -> str:
    """Envoie un message a l'agent avec conservation de la memoire."""
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        {"configurable": {"thread_id": thread_id}},
    )
    return response["messages"][-1].content


def run_demo():
    """Scenario de demonstration pour valider le TP."""
    agent = build_agent()

    prompts = [
        "Bonjour, je suis Khalid. Je n'aime pas le piment et je suis vegetarien.",
        "Memorise aussi que j'ai seulement 20 minutes pour cuisiner le soir.",
        "Dans mon refrigerateur j'ai des oeufs, tomates, fromage, oignons et pain. Propose-moi maintenant un ou deux plats adaptes a mes ingredients et mes preferences.",
        (
            "Donne-moi une variante differente. "
            "Si necessaire, utilise le web pour verifier une idee de recette."
        ),
    ]

    for index, prompt in enumerate(prompts, start=1):
        print(f"\n--- Question {index} ---")
        print(f"Utilisateur: {prompt}\n")
        answer = ask_agent(agent, prompt)
        print("Agent:")
        print(answer)


def run_interactive():
    """Mode interactif pour la demonstration en classe."""
    agent = build_agent()
    thread_id = "chef-thread-1"

    print("Agent Chef Personnel pret.")
    print("Tapez 'quit' pour terminer.\n")

    while True:
        user_text = input("Vous: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Fin de session.")
            break
        if not user_text:
            continue

        answer = ask_agent(agent, user_text, thread_id=thread_id)
        print(f"\nChef: {answer}\n")


if __name__ == "__main__":
    mode = os.getenv("APP_MODE", "interactive").lower()
    if mode == "interactive":
        run_interactive()
    else:
        run_demo()
