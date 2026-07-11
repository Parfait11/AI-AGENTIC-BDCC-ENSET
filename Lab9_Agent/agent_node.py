# PARTIE 2 : Agent comme nœud de LangGraph
#
# L'état contient :
#   - messages   : historique complet de la conversation (reducer `add`)
#   - llm_calls  : compteur du nombre d'appels au LLM
#
# llm_call laisse le LLM décider d'appeler un tool ou de répondre directement ;
# tool_node exécute les tool_calls demandés ; should_continue route vers
# tool_node tant que le LLM demande des outils, sinon termine le graphe.

from operator import add
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langchain.messages import SystemMessage, ToolMessage, HumanMessage

from langgraph.graph import StateGraph, START, END

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


def should_continue(state: AgentState) -> Literal["tool_node", END]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if last.tool_calls:
            return "tool_node"
    return END


builder = StateGraph(AgentState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()


if __name__ == "__main__":
    result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")], "llm_calls": 0})
    print(result)

    print("\n--- Historique conversationnel (Add 3 and 4) ---")
    for m in result["messages"]:
        try:
            m.pretty_print()
        except Exception as ex:
            print(ex)
            print(m)

    # Stream updates (state deltas)
    print("\n--- Stream 'updates' (Multiply 30 and 43) ---")
    for chunk in agent.stream(
        {"messages": [HumanMessage(content="Multiple 30 and 43.")], "llm_calls": 0},
        stream_mode="updates",
    ):
        print(chunk)

    # Stream messages (LLM tokens + métadonnées)
    print("\n--- Stream 'messages' (Divide 30 and 43) ---")
    for message_chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content="Divide 30 and 43.")], "llm_calls": 0},
        stream_mode="messages",
    ):
        if getattr(message_chunk, "content", None):
            print(message_chunk.content, end="", flush=True)
    print()
