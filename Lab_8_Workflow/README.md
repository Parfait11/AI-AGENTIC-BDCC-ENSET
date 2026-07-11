# LAB 8 : Workflows avec LangGraph

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Découvrir les briques de base d'un **workflow LangGraph** : graphe d'états, nodes, edges, reducers, état de type message, branchements conditionnels et boucles.

---

## Structure du projet

```
Lab8-Workflow_avec_LangGraph/
├── hello_graph.py                      # Partie 1 : premier graphe (StateGraph + MessagesState)
├── workflows/
│   └── two_step_workflow.py            # Partie 2 : workflow séquentiel à deux étapes
├── reducers_demo.py                    # Partie 3 : reducer (fusion de listes avec `add`)
├── message_state.py                    # Partie 4 : état de type message
├── conditional_workflow.py             # Partie 5 : branchement conditionnel
├── workflow_loop.py                    # Partie 6 : workflow en boucle + export PNG du graphe
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

## Concept : un Workflow LangGraph

```
StateGraph(State)
   │
   ├── add_node("nom", fonction)     # une étape du workflow
   ├── add_edge(START, "nom")        # liaison séquentielle
   ├── add_conditional_edges(...)    # branchement dynamique selon l'état
   │
   └── compile() ──► graph.invoke(state_initial)
```

Chaque node reçoit l'état courant et retourne un **dictionnaire de mise à jour** ; LangGraph fusionne ces mises à jour dans l'état global selon les règles définies (remplacement par défaut, ou **reducer** personnalisé).

---

## Partie 1 — Hello Graph (`hello_graph.py`)

Premier graphe minimal : un seul node `hello` qui ajoute un message assistant.

```python
def hello_node(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "Hello World"}]}

builder = StateGraph(MessagesState)
builder.add_node("hello", hello_node)
builder.add_edge(START, "hello")
builder.add_edge("hello", END)
```

```bash
uv run --active python hello_graph.py
```

**Résultat :**
```
Hello World
```

---

## Partie 2 — Workflow à deux étapes (`workflows/two_step_workflow.py`)

`refine_topic` transforme le sujet, puis `write_joke` génère une blague à partir du sujet affiné — exécution strictement séquentielle (`START → refine_topic → write_joke → END`).

```bash
uv run --active python workflows/two_step_workflow.py
```

**Résultat :**
```
{'topic': 'ice cream (and cats)', 'joke': 'Here is a joke about ice cream (and cats).'}
```

---

## Partie 3 — Reducer (`reducers_demo.py`)

Un **reducer** définit comment fusionner une nouvelle valeur avec l'ancienne dans le state, au lieu de l'écraser. Ici `log: Annotated[list[str], add]` : chaque retour de node est **concaténé** à la liste existante via `operator.add`.

```python
class State(TypedDict):
    topic: str
    log: Annotated[list[str], add]
```

```bash
uv run --active python reducers_demo.py
```

**Résultat :**
```
['step_a saw topic=langgraph', 'step_b finishing topic=langgraph', 'step_c finishing topic at last=langgraph']
```

Sans reducer, seul le dernier `log` retourné serait conservé ; avec `add`, l'historique complet des trois étapes est préservé.

---

## Partie 4 — État de type message (`message_state.py`)

`messages: Annotated[list[BaseMessage], add]` permet d'accumuler automatiquement l'historique conversationnel à chaque passage de node, exactement comme `log` dans la partie précédente — mais appliqué aux messages.

```bash
uv run --active python message_state.py
```

**Résultat :**
```
[{'role': 'user', 'content': 'hello'},
 {'role': 'ai', 'content': 'Step 1: got your message.'},
 {'role': 'ai', 'content': 'Step 2: got your message 1.'}]
```

Le message utilisateur initial est conservé, et chaque node (`echo`, `echo_1`) ajoute son propre message à la suite — le compteur `steps` s'incrémente à chaque étape.

---

## Partie 5 — Workflow conditionnel (`conditional_workflow.py`)

`check_joke` agit comme un **routeur conditionnel** : si la blague générée contient déjà un `"!"`, le graphe se termine directement ; sinon il passe par le node `improve_joke`.

```python
def check_joke(state: State) -> Literal["improve", "end"]:
    if "!" in state["joke"]:
        return "end"
    return "improve"

builder.add_conditional_edges("generate_joke", check_joke,
                              {"improve": "improve_joke", "end": END})
```

```bash
uv run --active python conditional_workflow.py
```

**Résultat :**
```
{'topic': 'cats', 'joke': 'A joke about cats maybe ? ', 'improved': 'A joke about cats maybe ?  !!!'}
```

La blague initiale ne contient pas de `"!"` → le graphe passe par `improve_joke`, qui ajoute `" !!!"` et alimente le champ `improved`.

---

## Partie 6 — Workflow en boucle (`workflow_loop.py`)

Le node `step` incrémente `n` et journalise chaque itération dans `log`. `should_continue` boucle sur `step` tant que `n < 5`, puis arrête le graphe (`END`).

```python
def should_continue(state: State) -> Literal["again", "stop"]:
    return "again" if state["n"] < 5 else "stop"

builder.add_conditional_edges("step", should_continue, {"again": "step", "stop": END})
```

```bash
uv run --active python workflow_loop.py
```

**Résultat :**
```
{'n': 5, 'log': ['n is now 1', 'n is now 2', 'n is now 3', 'n is now 4', 'n is now 5']}
Graphe exporté dans graph2.png
```

Le graphe boucle 5 fois sur `step` avant d'atteindre la condition d'arrêt, puis exporte une représentation visuelle du graphe (`graph2.png`) via `draw_mermaid_png()`.

---

## Récapitulatif des concepts

| Partie | Concept | Mécanisme clé |
|---|---|---|
| 1 | Graphe minimal | `StateGraph`, `add_node`, `add_edge`, `MessagesState` |
| 2 | Workflow séquentiel | Chaînage de plusieurs nodes via `add_edge` |
| 3 | Reducer | `Annotated[list[...], add]` — fusion au lieu d'écrasement |
| 4 | État de type message | Accumulation automatique de l'historique conversationnel |
| 5 | Branchement conditionnel | `add_conditional_edges` + fonction routeur `Literal[...]` |
| 6 | Boucle | Un node qui se renvoie à lui-même via une condition d'arrêt |
