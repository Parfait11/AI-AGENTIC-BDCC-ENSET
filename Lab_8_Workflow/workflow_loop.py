# PARTIE 6 : Workflow en boucle
# `should_continue` renvoie au node "step" tant que n < 5 (boucle),
# puis arrête le graphe (END) une fois la condition atteinte.

from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    n: int
    log: list[str]


def step(state: State):
    n = state["n"] + 1
    return {"n": n, "log": state["log"] + [f"n is now {n}"]}


def should_continue(state: State) -> Literal["again", "stop"]:
    return "again" if state["n"] < 5 else "stop"


builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_conditional_edges("step", should_continue, {"again": "step", "stop": END})

graph = builder.compile()

result = graph.invoke({"n": 0, "log": []})
print(result)

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph2.png", "wb") as f:
        f.write(png_bytes)
    print("Graphe exporté dans graph2.png")
except Exception as e:
    print(f"Génération du PNG ignorée (pas de connexion au service mermaid.ink) : {e}")
