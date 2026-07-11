import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool, ToolRuntime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# ============================================================
# RAG — indexation de la base de recettes locale
# ============================================================

RECIPES_FILE = Path(__file__).parent / "recipes.txt"
recipes_text = RECIPES_FILE.read_text(encoding="utf-8")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.create_documents([recipes_text])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# ============================================================
# État personnalisé
# ============================================================

class ChefState(AgentState):
    preferences: list[str]


# ============================================================
# Outils
# ============================================================

@tool
def search_recipes_rag(query: str) -> str:
    """Search the local recipe knowledge base for dishes, ingredients, and culinary techniques."""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No matching recipes found in the knowledge base."
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool
def search_web(query: str) -> str:
    """Search the web for recipes, culinary techniques, and ingredient combinations."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return (
            "Web search not available (TAVILY_API_KEY not configured). "
            "Using the local recipe knowledge base instead."
        )
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke({"query": query})
        return "\n\n".join(r.get("content", "") for r in results)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def remember_preference(preference: str, runtime: ToolRuntime) -> Command:
    """Store a user preference (dietary restriction, allergy, favourite cuisine) in long-term memory."""
    try:
        current = list(runtime.state["preferences"])
    except KeyError:
        current = []
    updated = current + [preference]
    return Command(update={
        "preferences": updated,
        "messages": [ToolMessage(
            f"Preference saved: {preference}",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def get_preferences(runtime: ToolRuntime) -> str:
    """Retrieve all stored user preferences, dietary restrictions, and allergies."""
    try:
        preferences = runtime.state["preferences"]
    except KeyError:
        return "No preferences stored yet."
    if not preferences:
        return "No preferences stored yet."
    return "User preferences: " + " | ".join(preferences)


# ============================================================
# Agent chef cuisinier
# ============================================================

SYSTEM_PROMPT = """You are a personal chef assistant. Your role is to help users discover
delicious dishes they can prepare with their available ingredients.

You have access to:
- search_recipes_rag: search your local recipe knowledge base
- search_web: search the internet for additional recipes and techniques
- remember_preference: store user preferences, allergies, and dietary restrictions
- get_preferences: retrieve the user's stored preferences

When a user tells you their available ingredients:
1. First check their preferences with get_preferences
2. Search the recipe knowledge base with search_recipes_rag
3. Suggest 2-3 concrete, detailed dishes adapted to their ingredients and preferences
4. Always respect their dietary restrictions and allergies

When a user shares a preference, allergy, or dietary restriction, save it with remember_preference.

Be enthusiastic, helpful, and creative like a professional chef!"""

model = ChatOllama(model="llama3.2:3b", temperature=0)

chef_agent = create_agent(
    model=model,
    tools=[search_recipes_rag, search_web, remember_preference, get_preferences],
    state_schema=ChefState,
    checkpointer=InMemorySaver(),
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "chef_demo"}}

    def chat(message: str):
        response = chef_agent.invoke(
            {"messages": [HumanMessage(content=message)], "preferences": []},
            config,
        )
        print(f"\nUser  : {message}")
        print(f"Chef  : {response['messages'][-1].content}")

    print("=" * 60)
    print("DEMO : Agent Chef Cuisinier Personnel")
    print("=" * 60)

    # Enregistrer les preferences
    chat("Je suis vegetarien et j'adore la cuisine italienne.")
    chat("Je suis allergique aux arachides.")

    # Suggestions avec les ingredients disponibles
    chat(
        "J'ai dans mon frigo : des pates, des tomates, de l'ail, "
        "de l'huile d'olive, du basilic et du parmesan. "
        "Qu'est-ce que je peux cuisiner ?"
    )

    # Suivi avec de nouveaux ingredients
    chat("Et si j'ajoute des oeufs et des epinards, qu'est-ce que je peux faire ?")
