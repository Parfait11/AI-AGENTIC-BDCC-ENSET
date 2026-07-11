# LAB 9 : Agent avec LangGraph

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Construire progressivement un **agent LangGraph** complet : un LLM doté d'outils (tools), exécuté comme nœud de graphe, capable de s'arrêter pour validation humaine (HITL), de reprendre après interruption, de conserver son historique de checkpoints et de "forker" (revenir dans le passé et relancer depuis un état modifié).

---

## Structure du projet

```
Lab9-Agent_avec_LangGraph/
├── tools_setup.py        # Partie 1 : LLM local (Ollama) + tools arithmétiques
├── agent_node.py         # Partie 2 : agent comme nœud de graphe (llm_call / tool_node)
├── hitl_workflow.py      # Partie 3 : workflow @entrypoint/@task avec interrupt() (HITL)
├── tp_advanced_agent.py  # Partie 4 : TP — agent complet (tools + HITL + historique + fork)
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

## Partie 1 — LLM local avec outils (`tools_setup.py`)

Trois tools arithmétiques (`add`, `multiply`, `divide`) sont liés au modèle Ollama via `bind_tools` :

```python
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)
```

Ce module est importé par les parties suivantes — il ne s'exécute pas seul.

---

## Partie 2 — Agent comme nœud de LangGraph (`agent_node.py`)

L'état (`AgentState`) contient :
- `messages` : historique complet, fusionné automatiquement via le reducer `add`
- `llm_calls` : compteur d'appels au LLM, incrémenté à chaque passage par `llm_call`

```
START ──► llm_call ──► should_continue ──┬──► tool_node ──► llm_call (boucle)
                                          └──► END
```

- `llm_call` : le LLM décide d'appeler un tool ou de répondre directement
- `tool_node` : exécute les `tool_calls` demandés et retourne des `ToolMessage`
- `should_continue` : route vers `tool_node` tant que le dernier message contient des `tool_calls`, sinon `END`

```bash
uv run --active python agent_node.py
```

### Résultats obtenus

```
--- Historique conversationnel (Add 3 and 4) ---
================================ Human Message =================================
Add 3 and 4.
================================== Ai Message ==================================
Tool Calls:
  add (...)
    a: 3
    b: 4
================================= Tool Message =================================
7
================================== Ai Message ==================================
The result of adding 3 and 4 is 7.

--- Stream 'updates' (Multiply 30 and 43) ---
{'llm_call': {'messages': [AIMessage(..., tool_calls=[{'name': 'multiply', 'args': {'a': '30', 'b': '43'}, ...}])], 'llm_calls': 1}}
{'tool_node': {'messages': [ToolMessage(content='1290', ...)]}}
{'llm_call': {'messages': [AIMessage(content='The result of multiplying 30 by 43 is 1290.', ...)], 'llm_calls': 2}}

--- Stream 'messages' (Divide 30 and 43) ---
0.6976744186046512The result of dividing 30 by 43 is approximately 0.6976.
```

**Constat :** `stream_mode="updates"` affiche chaque modification du graphe nœud par nœud (le LLM décide d'appeler `multiply`, le tool répond `1290`, puis le LLM formule la réponse finale). `stream_mode="messages"` diffuse les tokens du LLM en direct — on voit d'abord le résultat brut du tool (`0.69767...`) streamé par le modèle, puis sa phrase de synthèse.

---

## Partie 3 — Workflow HITL avec `@entrypoint` / `@task` (`hitl_workflow.py`)

- `@task` transforme une fonction Python en brique d'exécution isolée et asynchrone (`.result()` attend le résultat)
- `@entrypoint` définit le point d'entrée du workflow : il orchestre les tasks, peut appeler `interrupt()` pour suspendre l'exécution, et reprend via `Command(resume=...)`

```python
@task
def write_essay(topic: str) -> str:
    time.sleep(1)
    return f"Essay draft about {topic}"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    draft = write_essay(topic).result()
    approved = interrupt({"draft": draft, "action": "approve or reject"})
    return {"draft": draft, "approved": approved}
```

```bash
uv run --active python hitl_workflow.py
```

### Résultats obtenus

```
--- Première exécution (génère le brouillon, puis interrupt) ---
{'write_essay': 'Essay draft about cats'}
{'__interrupt__': (Interrupt(value={'draft': 'Essay draft about cats', 'action': 'approve or reject'}, ...),)}

--- Deuxième exécution (reprise après validation humaine) ---
{'workflow': {'draft': 'Essay draft about cats', 'approved': True}}
```

**Constat :** la première exécution génère le brouillon puis **suspend** le graphe via `interrupt()` — aucune réponse finale n'est encore produite. La seconde exécution, avec le **même** `thread_id` et `Command(resume=True)`, reprend exactement où le graphe s'était arrêté et retourne le résultat final avec `approved: True`.

---

## Partie 4 — TP : Agent avancé (`tp_advanced_agent.py`)

Combine tous les concepts précédents dans un seul agent :

```
START ──► llm_call ──► should_continue ──┬──► approve ──► interrupt() ──┬──► tool_node ──► llm_call
                                          └──► END                       └──► END (si rejeté)
```

- `approve_node` appelle `interrupt()` pour demander une validation humaine **avant** d'exécuter le tool, puis route via `Command(goto="tool_node" if decision else END)`
- `agent.get_state(config)` / `get_state_history(config)` permettent d'inspecter les checkpoints sauvegardés
- `agent.update_state(...)` + `agent.invoke(None, new_config)` permettent de **forker** : repartir d'un checkpoint passé avec un état modifié

```bash
uv run --active python tp_advanced_agent.py
```

### Résultats obtenus

```
--- Scénario 1 : interruption avant exécution du tool ---
Interrupt payload: {'question': 'Approve tool execution?',
                    'tool_calls': [{'name': 'add', 'args': {'a': '3', 'b': '4'}, ...}]}
Done. Last message: content='The result of adding 3 and 4 is 7.'
Latest checkpoint next: ()
Number of checkpoints: 6
Most recent checkpoint id: 1f162dd2-2aab-6415-8004-57e8a11e62c2

--- Scénario 2 : rejet de l'exécution du tool ---
Interrupt payload: {'question': 'Approve tool execution?',
                    'tool_calls': [{'name': 'multiply', 'args': {'a': '30', 'b': '41'}, ...}]}
Done. Last message: content='' tool_calls=[{'name': 'multiply', 'args': {'a': '30', 'b': '41'}, ...}]
Forked: content='Multiply 30 and 41.'
```

**Constat :**
- **Scénario 1 (approbation)** : le graphe s'arrête avant `tool_node`, demande une validation (`Approve tool execution?`), puis `Command(resume=True)` laisse l'exécution continuer normalement jusqu'à la réponse finale (`7`). Six checkpoints sont sauvegardés pour ce thread.
- **Scénario 2 (rejet + fork)** : `Command(resume=False)` fait router `approve_node` directement vers `END` — le tool **n'est jamais exécuté**, le dernier message reste l'`AIMessage` contenant le `tool_call` non exécuté. En sélectionnant un ancien checkpoint (`history[1]`, juste après `START`) et en y injectant un nouvel état via `update_state`, `agent.invoke(None, new_config)` **relance le graphe depuis ce point historique** — démontrant la capacité de LangGraph à "voyager dans le temps" et explorer des branches alternatives d'exécution.

---

## Récapitulatif des concepts

| Partie | Concept | Mécanisme clé |
|---|---|---|
| 1 | LLM + tools | `@tool`, `model.bind_tools(tools)` |
| 2 | Agent comme nœud | `llm_call` / `tool_node` / `should_continue`, reducer `add` sur `messages` |
| 3 | HITL fonctionnel | `@entrypoint`, `@task`, `interrupt()`, `Command(resume=...)` |
| 4 | Agent avancé complet | `approve_node` + `Command(goto=...)`, `get_state_history`, `update_state` (fork) |
