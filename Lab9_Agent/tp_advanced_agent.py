# PARTIE 4 : TP — Agent LangGraph avancé
#
# Cet agent est capable de :
#   - utiliser des tools (add, multiply, divide)
#   - s'arrêter pour validation humaine avant d'exécuter un tool (HITL)
#   - reprendre après interruption via Command(resume=...)
#   - sauvegarder l'historique complet des checkpoints
#   - revenir dans le passé et modifier un état pour relancer le graphe (fork)

from operator import add
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langchain.messages import SystemMessage, ToolMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from tools_setup import model_with_tools, tools_by_name


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    llm_calls: int


def llm_call(state: AgentState):
    """LLM decides whether to call tools or respond."""
    response = model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant that solves arithmetic problem using tools when needed.")]
        + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: AgentState):
    """Execute tool calls from last AI message."""
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        tool = tools_by_name[call["name"]]
        observation = tool.invoke(call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["approve", END]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and last.tool_calls:
        return "approve"
    return END


def approve_node(state: AgentState) -> Command[Literal["tool_node", "__end__"]]:
    """Interrompt le graphe pour demander à l'humain s'il valide l'appel au tool."""
    decision = interrupt({"question": "Approve tool execution?", "tool_calls": state["messages"][-1].tool_calls})
    return Command(goto="tool_node" if decision else END)


builder = StateGraph(AgentState)

builder.add_node("llm_call", llm_call)
builder.add_node("approve", approve_node)
builder.add_node("tool_node", tool_node)

checkpointer = InMemorySaver()

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["approve", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile(checkpointer=checkpointer)

# ============================================================
# Scénario 1 : approbation (resume=True)
# ============================================================
config = {"configurable": {"thread_id": "thread-1"}}

result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")], "llm_calls": 0}, config=config)

print("--- Scénario 1 : interruption avant exécution du tool ---")
print("Interrupt payload:", result["__interrupt__"][0].value)

resume = agent.invoke(Command(resume=True), config=config)
print("Done. Last message:", resume["messages"][-1])

snap = agent.get_state(config)
print("Latest checkpoint next:", snap.next)

history = list(agent.get_state_history(config))
print("Number of checkpoints:", len(history))
print("Most recent checkpoint id:", history[0].config["configurable"]["checkpoint_id"])

# ============================================================
# Scénario 2 : rejet (resume=False) puis fork dans l'historique
# ============================================================
config_reject = {"configurable": {"thread_id": "thread-1-reject"}}

result = agent.invoke({"messages": [HumanMessage(content="Multiply 30 and 41.")], "llm_calls": 0}, config=config_reject)

print("\n--- Scénario 2 : rejet de l'exécution du tool ---")
print("Interrupt payload:", result["__interrupt__"][0].value)

resume = agent.invoke(Command(resume=False), config=config_reject)
print("Done. Last message:", resume["messages"][-1])

history = list(agent.get_state_history(config_reject))
picked = history[1]

new_config = agent.update_state(
    picked.config,
    values={"messages": [HumanMessage(content="Multiply 30 and 41.")], "llm_calls": 0},
)
forked = agent.invoke(None, new_config)  # relance le graphe depuis un point historique
print("Forked:", forked["messages"][-1])
