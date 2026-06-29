from typing import Literal

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import MAX_REWRITES
from src.llm import get_llm
from src.state import AgentState
from src.tools import TOOLS, TOOLS_BY_NAME

SYSTEM_PROMPT = (
    "Tu es un assistant specialise dans la gestion des epidemies de cholera : "
    "surveillance epidemiologique, reponse de terrain, rehydratation orale (SRO), "
    "prevention et controle. "
    "Utilise l'outil retrieve_documents pour t'appuyer sur les guides OMS, GTFCC et MSF "
    "avant de repondre a une question de connaissance generale. "
    "Utilise compute_ors_dosage pour calculer les doses de SRO selon le poids et le niveau "
    "de deshydratation du patient. "
    "Utilise estimate_attack_rate pour calculer le taux d'attaque d'une epidemie dans une zone donnee. "
    "Reponds en francais, de facon claire et precise, et cite tes sources "
    "quand tu t'appuies sur un document."
)

model = get_llm()
model_with_tools = model.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Le LLM decide d'appeler un outil ou de repondre directement."""
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


def tools_node(state: AgentState) -> dict:
    """Execute les tool calls demandes par le dernier message de l'agent."""
    last = state["messages"][-1]
    results = []
    tool_names = []
    retrieved_docs = state.get("retrieved_docs", [])
    for call in last.tool_calls:
        tool = TOOLS_BY_NAME[call["name"]]
        try:
            observation = tool.invoke(call["args"])
        except Exception as exc:
            observation = (
                f"Erreur : l'appel a l'outil {call['name']} a echoue ({exc}). "
                "Verifie les arguments et reessaie, ou reponds sans cet outil."
            )
        results.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
        tool_names.append(call["name"])
        if call["name"] == "retrieve_documents":
            retrieved_docs = [str(observation)]
    return {"messages": results, "last_tool_names": tool_names, "retrieved_docs": retrieved_docs}


def grade_documents_node(state: AgentState) -> dict:
    """Le LLM evalue si les documents recuperes sont pertinents pour la question."""
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )
    docs_text = "\n\n".join(state.get("retrieved_docs", []))
    prompt = (
        f"Question : {question}\n\nDocuments recuperes :\n{docs_text}\n\n"
        "Ces documents contiennent-ils des informations utiles pour repondre a la "
        "question ? Reponds uniquement par 'oui' ou 'non'."
    )
    response = model.invoke([HumanMessage(content=prompt)])
    relevant = "oui" in response.content.strip().lower()
    return {"docs_relevant": relevant}


def rewrite_query_node(state: AgentState) -> dict:
    """Reformule la question utilisateur quand les documents recuperes ne sont pas pertinents."""
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )
    prompt = (
        f"La question suivante n'a pas permis de trouver des documents pertinents : "
        f"'{question}'. Reformule-la avec des termes differents ou plus generaux, "
        "en gardant le meme sens. Reponds uniquement avec la question reformulee."
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [HumanMessage(content=response.content.strip())],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def route_after_tools(state: AgentState) -> Literal["grade_documents", "agent"]:
    if "retrieve_documents" in state.get("last_tool_names", []):
        return "grade_documents"
    return "agent"


def route_after_grading(state: AgentState) -> Literal["agent", "rewrite_query"]:
    if state.get("docs_relevant", True) or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "agent"
    return "rewrite_query"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("rewrite_query", rewrite_query_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
builder.add_conditional_edges("tools", route_after_tools, ["grade_documents", "agent"])
builder.add_conditional_edges("grade_documents", route_after_grading, ["agent", "rewrite_query"])
builder.add_edge("rewrite_query", "agent")

graph = builder.compile(checkpointer=InMemorySaver())