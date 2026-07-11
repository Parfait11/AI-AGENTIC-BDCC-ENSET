# TP : Multi-Agents en LangChain
#
# Architecture hiérarchique :
#   main_agent ──tool──► call_subagent_1 ──► subagent_1 (square_root)
#              └─tool──► call_subagent_2 ──► subagent_2 (square)
#
# Chaque sous-agent est un agent LangChain à part entière, exposé au
# main_agent comme un simple @tool — le main_agent décide lui-même
# quel sous-agent appeler en fonction de la question posée.

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:3b", temperature=0)

# ============================================================
# PARTIE 1 : Définition des outils
# ============================================================


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x**0.5


@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x**2


# ============================================================
# PARTIE 2 : Création des sous-agents
# ============================================================

subagent_1 = create_agent(model=model, tools=[square_root])
subagent_2 = create_agent(model=model, tools=[square])


# ============================================================
# PARTIE 3 : Créer l'agent principal
# ============================================================


@tool
def call_subagent_1(x: float) -> str:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content


@tool
def call_subagent_2(x: float) -> str:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content


main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt=(
        "You are a helpful assistant who can call subagents to calculate "
        "the square root or square of a number. Always give the EXACT "
        "numerical result returned by the subagent."
    ),
)


# ============================================================
# PARTIE 4 : Appeler les agents et afficher le résultat
# ============================================================

if __name__ == "__main__":
    question_1 = "What is the square root of 456?"
    response_1 = main_agent.invoke({"messages": [HumanMessage(content=question_1)]})
    print(f"Q: {question_1}")
    print(f"R: {response_1['messages'][-1].content}\n")

    question_2 = "What is the square of 12?"
    response_2 = main_agent.invoke({"messages": [HumanMessage(content=question_2)]})
    print(f"Q: {question_2}")
    print(f"R: {response_2['messages'][-1].content}")
