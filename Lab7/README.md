# LAB 7 : Un agent HITL (Human-In-The-Loop)

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Implémenter un agent IA dont l'exécution peut être **interrompue** pour demander une validation humaine avant de continuer — en utilisant `interrupt()` de LangGraph.

---

## Structure du projet

```
Lab7-Human_In_The_Loop/
├── agent_hitl.py    # Les 5 parties HITL
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Prérequis

- Python >= 3.10 · [uv](https://docs.astral.sh/uv/) · [Ollama](https://ollama.com/) avec `llama3.2:3b`

```bash
uv sync
```

---

## Concept HITL

Un agent **Human-In-The-Loop** s'arrête à des points clés pour demander une décision humaine avant de continuer.

```
Agent exécute             Humain décide           Agent continue
─────────────             ─────────────           ──────────────
read_email()    ───────►  ◄─ interrupt() ─►  ───► approve / reject / edit
send_email()    bloqué    décision requise        résume avec Command(...)
```

---

## Partie 1 — Définition des outils

```python
@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the inbox."""
    return runtime.state["email"]   # lit depuis l'état de l'agent

@tool
def send_email(body: str) -> str:
    """Send an email. Requires human approval before sending."""
    decision = interrupt({
        "action_requests": [{"name": "send_email", "args": {"body": body}}]
    })
    d = decision["decisions"][0]
    if d["type"] == "approve":
        return "Email sent successfully."
    elif d["type"] == "reject":
        return f"Email rejected: {d.get('message', '')}"
    elif d["type"] == "edit":
        return f"Email sent (modified): {d['edited_action']['args']['body']}"
```

`interrupt()` de `langgraph.types` suspend le graph et retourne la valeur de reprise passée via `Command(resume=...)`.

---

## Partie 2 — Création et invocation initiale

```bash
uv run --active python agent_hitl.py
```

```python
class EmailState(AgentState):
    email: str

agent = create_agent(
    model=ChatOllama(model="llama3.2:3b"),
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
)

response = agent.invoke(
    {"messages": [HumanMessage(content="Lisez et répondez à mon e-mail.")],
     "email": "Bonjour Sara, je vais être en retard..."},
    {"configurable": {"thread_id": "1"}}
)
```

L'agent lit le mail, rédige une réponse, puis **s'arrête** avant d'envoyer :

```python
print(response['__interrupt__'])
# [Interrupt(value={'action_requests': [{'name': 'send_email', 'args': {'body': '...'}}]}, ...)]

print(response['__interrupt__'][0].value['action_requests'][0]['args']['body'])
# Corps du mail que l'agent veut envoyer
```

---

## Partie 3 — Approuver

```python
from langgraph.types import Command

response = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config  # même thread_id pour reprendre
)
print(response['messages'][-1].content)
# Email sent successfully.
```

---

## Partie 4 — Refuser

```python
response = agent.invoke(
    Command(resume={
        "decisions": [{"type": "reject", "message": "J'annule notre rendez-vous."}]
    }),
    config
)
print(response)
# send_email retourne : "Email rejected: J'annule notre rendez-vous."
# Le LLM gère le rejet et adapte sa réponse
```

---

## Partie 5 — Modifier

```python
response = agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "send_email",
                "args": {"body": "Je suis désolée mais je dois annuler notre rendez-vous. Sara"}
            }
        }]
    }),
    config
)
print(response['messages'][-1].content)
# Email sent (modified): Je suis désolée...
```

---

## Les 3 décisions HITL

| Type | Comportement | Paramètre supplémentaire |
|---|---|---|
| `"approve"` | Envoie le mail tel quel | — |
| `"reject"` | Annule l'envoi, retourne la raison au LLM | `"message": "..."` |
| `"edit"` | Remplace les arguments et envoie le mail modifié | `"edited_action": {"name": ..., "args": {...}}` |

---

## Résultats obtenus

```
PARTIE 2 : Invocation initiale — interruption avant envoi
[Interrupt(value={'action_requests': [{'name': 'send_email', 'args': {'body': '...'}}]}, ...)]
Corps du mail proposé par l'agent: Répondre immédiatement...

PARTIE 3 : Approuver
"Répondre immédiatement, je vais répondre maintenant dans le même fil."

PARTIE 4 : Refuser
ToolMessage: "Email rejected: J'annule notre rendez-vous."
LLM: "Désolé(e), mais il semble que l'email a été rejeté..."

PARTIE 5 : Modifier
"Email sent (modified): Je suis désolée mais je dois annuler..."
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   LangGraph                     │
│                                                 │
│  HumanMessage ──► LLM ──► tool_calls            │
│                              │                  │
│                    ┌─────────┴────────┐          │
│                    │                 │          │
│               read_email()     send_email()     │
│               (lit l'état)   interrupt() ◄──┐  │
│                                    │         │  │
│                              ┌─────▼──────┐  │  │
│                              │  SUSPENDU  │  │  │
│                              │  Humain    │  │  │
│                              │  décide    │  │  │
│                              └─────┬──────┘  │  │
│                                    │         │  │
│                         Command(resume=...) ─┘  │
│                                    │            │
│                              approve/reject/edit │
│                                    │            │
│                         AIMessage (réponse) ◄───┘
└─────────────────────────────────────────────────┘
```
