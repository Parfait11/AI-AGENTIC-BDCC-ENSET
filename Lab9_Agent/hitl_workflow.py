# PARTIE 3 : Un workflow LangGraph qui génère un texte
#            et s'arrête pour validation humaine (HITL)
#
# @task    transforme une fonction Python en brique d'exécution isolée
# @entrypoint définit la fonction principale qui orchestre les tasks,
#             peut inclure interrupt() et se relance via Command(resume=...)

import time
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command


@task
def write_essay(topic: str) -> str:
    time.sleep(1)
    return f"Essay draft about {topic}"


@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    draft = write_essay(topic).result()
    approved = interrupt({"draft": draft, "action": "approve or reject"})
    return {"draft": draft, "approved": approved}


thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print("--- Première exécution (génère le brouillon, puis interrupt) ---")
for item in workflow.stream("cats", config):
    print(item)

print("\n--- Deuxième exécution (reprise après validation humaine) ---")
for item in workflow.stream(Command(resume=True), config):
    print(item)
