from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# ============================================================
# PARTIE 5 : État — données mutables qui évoluent et persistent
#            pendant la conversation (contrairement au contexte)
# ============================================================
# On étend `AgentState` pour ajouter un champ personnalisé : le genre
# littéraire favori du lecteur, révélé au fil de la discussion.


class LibraryState(AgentState):
    favourite_genre: str


model = ChatOllama(model="llama3.2:3b", temperature=0)

# ============================================================
# PARTIE 6 : Un tool qui modifie l'état via Command(update=...)
# ============================================================
# `Command` permet à un tool de mettre à jour l'état du graph (et pas
# seulement de retourner un message au LLM).


@tool
def remember_favourite_genre(genre: str, runtime: ToolRuntime) -> Command:
    """Store the reader's favourite literary genre once they reveal it."""
    return Command(
        update={
            "favourite_genre": genre,
            "messages": [
                ToolMessage(
                    f"Genre favori enregistré : {genre}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def recall_favourite_genre(runtime: ToolRuntime) -> str:
    """Read the reader's favourite literary genre from the persisted state."""
    try:
        return runtime.state["favourite_genre"]
    except KeyError:
        return "Aucun genre favori enregistré pour le moment."


agent = create_agent(
    model=model,
    tools=[remember_favourite_genre, recall_favourite_genre],
    state_schema=LibraryState,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "reader-1"}}

# ----- Tour 1 : le lecteur révèle son genre favori -----
response = agent.invoke(
    {"messages": [HumanMessage(content="J'adore les romans de science-fiction.")]},
    config,
)
print("--- Partie 6 : enregistrer le genre favori ---")
print(response["messages"][-1].content)
print("État courant :", response.get("favourite_genre"))

# ----- Tour 2 : même thread → l'état est toujours là -----
response = agent.invoke(
    {"messages": [HumanMessage(content="Rappelle-moi quel est mon genre littéraire favori.")]},
    config,
)
print("\n--- Partie 6 (suite) : récupérer le genre persisté ---")
print(response["messages"][-1].content)

# ----- Tour 3 : nouveau thread → l'état repart de zéro -----
response = agent.invoke(
    {"messages": [HumanMessage(content="Quel est mon genre littéraire favori ?")]},
    {"configurable": {"thread_id": "reader-2"}},
)
print("\n--- Partie 7 : nouveau thread, état vide ---")
print(response["messages"][-1].content)
