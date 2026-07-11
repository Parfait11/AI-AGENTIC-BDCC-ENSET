from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama

# ============================================================
# PARTIE 1 : Contexte — données immuables fournies à l'invocation
# ============================================================
# Le contexte représente des informations connues AVANT la conversation
# (profil utilisateur, paramètres de session...). Il est passé une fois
# via `context=` et reste identique pendant toute l'invocation.


@dataclass
class ReaderProfile:
    name: str = "Khalid"
    membership: str = "standard"
    preferred_language: str = "français"


model = ChatOllama(model="llama3.2:3b", temperature=0)

# ============================================================
# PARTIE 2 : Agent sans accès au contexte
# ============================================================
# Même si `context_schema` est déclaré, l'agent n'a aucun tool pour lire
# le contexte : le LLM ne peut donc pas répondre avec ces informations.

agent_no_access = create_agent(model=model, context_schema=ReaderProfile)

response = agent_no_access.invoke(
    {"messages": [HumanMessage(content="Quel est mon niveau d'abonnement à la bibliothèque ?")]},
    context=ReaderProfile(),
)
print("--- Partie 2 : Agent sans accès au contexte ---")
print(response["messages"][-1].content)


# ============================================================
# PARTIE 3 : Agent avec des tools qui lisent le contexte
# ============================================================
# `runtime.context` permet à un tool d'accéder aux données du contexte
# courant — ici le profil du lecteur de la bibliothèque.


@tool
def get_membership_level(runtime: ToolRuntime) -> str:
    """Get the library membership level of the current reader."""
    return runtime.context.membership


@tool
def get_preferred_language(runtime: ToolRuntime) -> str:
    """Get the preferred reading language of the current reader."""
    return runtime.context.preferred_language


agent_with_context = create_agent(
    model=model,
    tools=[get_membership_level, get_preferred_language],
    context_schema=ReaderProfile,
)

response = agent_with_context.invoke(
    {"messages": [HumanMessage(content="Quel est mon niveau d'abonnement à la bibliothèque ?")]},
    context=ReaderProfile(),
)
print("\n--- Partie 3 : Agent avec accès au contexte ---")
print(response["messages"][-1].content)


# ============================================================
# PARTIE 4 : Changer le contexte d'une invocation à l'autre
# ============================================================
# Le contexte n'est pas persisté : chaque appel à `invoke` peut fournir
# un contexte différent (ex : un autre lecteur, un autre abonnement).

response = agent_with_context.invoke(
    {"messages": [HumanMessage(content="Quel est mon niveau d'abonnement à la bibliothèque ?")]},
    context=ReaderProfile(name="Sara", membership="premium", preferred_language="anglais"),
)
print("\n--- Partie 4 : Changement de contexte (lecteur premium) ---")
print(response["messages"][-1].content)
