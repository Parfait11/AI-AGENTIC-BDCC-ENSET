# LAB 5 : LangGraph Studio & Multi-Agents

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Ce dossier regroupe deux TP complémentaires basés sur LangGraph Studio :

1. **Visualiser, tester et déboguer un agent** avec LangGraph Studio — visualiser le graphe d'un agent, l'exécuter étape par étape, inspecter les inputs/outputs de chaque node ;
2. **Construire un système multi-agents hiérarchique** — un agent principal qui délègue des sous-tâches à des sous-agents spécialisés, exposés comme de simples outils, et observer le tout dans Studio.

---

## Structure du projet

```
Lab5-LangGraph_Studio/
├── agent_simple.py      # TP "5 LangGraph Studio" : agent avec outil RAG simulé
├── multi_agents.py      # TP "8 Multi-Agents" : agent principal + 2 sous-agents
├── langgraph.json       # Configuration LangGraph Studio (3 graphes exposés)
├── pyproject.toml       # Dépendances uv
├── .env.example         # Template variables d'environnement
└── .gitignore
```

---

## Prérequis

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) installé
- [Ollama](https://ollama.com/) avec `llama3.2:3b` (`ollama pull llama3.2:3b`)
- Compte [LangSmith](https://smith.langchain.com/) avec une clé API

### Installation

```bash
# Copier et remplir le fichier d'environnement
cp .env.example .env
# → Ajouter au minimum : LANGSMITH_API_KEY

# Installer les dépendances
uv sync
```

---

## Partie 1 — Créer un compte LangSmith et importer une clé

1. Aller sur **https://smith.langchain.com/**
2. Se connecter / créer un compte (Google, GitHub ou Email)
3. Aller dans **Settings → API Keys → Create API Key**
4. Donner un nom (ex: `langgraph-project`) et copier la clé
5. Ajouter dans `.env` :
   ```
   LANGSMITH_API_KEY=lsv2_pt_...
   ```

---

## Partie 2 — Créer l'agent

Le fichier `agent_simple.py` définit un agent LangChain avec un outil RAG simulé :

```python
@tool
def rag_search_opt(query: str) -> str:
    """Recherche des informations dans le texte."""
    return "Le personnage principal est un jeune homme nommé Jack..."

agent = create_agent(
    model=ChatOllama(model="llama3.2:3b", temperature=0),
    tools=[rag_search_opt],
    system_prompt="Tu es un assistant spécialisé dans l'analyse de texte..."
)
```

**Test rapide :**
```bash
uv run --active python -c "
from langchain.messages import HumanMessage
from agent_simple import agent
r = agent.invoke({'messages': [HumanMessage(content='Qui est le personnage principal ?')]})
print(r['messages'][-1].content)
"
```

---

## Partie 2 — Fichier de configuration `langgraph.json`

Ce fichier indique à LangGraph Studio où trouver l'agent, quel environnement utiliser, et comment lancer le projet :

```json
{
    "graphs": {
        "agent_simple": "./agent_simple.py:agent"
    },
    "env": "./.env",
    "source": {
        "kind": "uv",
        "root": "."
    }
}
```

| Clé | Description |
|---|---|
| `graphs` | Nom du graphe → chemin du fichier : variable |
| `env` | Fichier `.env` à charger |
| `source.kind` | Gestionnaire de paquets (`uv`) |
| `source.root` | Répertoire racine du projet |

---

## Partie 2 — Lancer LangGraph Studio

```bash
uv run --active langgraph dev
```

Le serveur démarre sur **http://127.0.0.1:2024** et ouvre automatiquement l'interface Studio dans le navigateur via **https://smith.langchain.com/studio**.

### Ce que vous pouvez faire dans Studio

| Fonctionnalité | Description |
|---|---|
| **Graph** | Visualise le graphe : `__start__ → model ⇄ tools → __end__` |
| **Chat** | Envoie des messages et voit les réponses en temps réel |
| **Interrupts** | Pause l'exécution à un node pour inspecter l'état |
| **Memory** | Visualise l'état persistant du thread |
| **Tracing** | Inspecte chaque appel LLM et tool call |

### Graphe de l'agent

```
        __start__
            │
            ▼
          model  ◄────┐
            │         │
     ┌──────┴──────┐  │
     │             │  │
   __end__       tools─┘
```

---

## Résultat attendu

Après `langgraph dev`, l'interface Studio affiche :

- Le graphe de l'agent avec les nodes `model` et `tools`
- Un panneau **Input** pour envoyer des messages
- La trace complète de chaque exécution (LLM call → tool call → réponse)

**Exemple d'interaction :**
> **User:** Qui est le personnage principal ?
> **Agent:** *(appelle `rag_search_opt`)* → Le personnage principal est Jack, un jeune homme qui découvre un ancien artefact magique.

---

# TP : Multi-Agents en LangChain (`multi_agents.py`)

## Architecture hiérarchique

```
                         main_agent
                  (call_subagent_1 / call_subagent_2)
                    │                         │
                    ▼                         ▼
              call_subagent_1           call_subagent_2
                    │                         │
                    ▼                         ▼
               subagent_1                subagent_2
              tools=[square_root]       tools=[square]
```

Chaque sous-agent est un agent LangChain complet et autonome (avec son propre LLM et son propre outil), mais il est exposé au `main_agent` comme un simple `@tool`. Le `main_agent` décide lui-même, en fonction de la question posée, quel sous-agent appeler.

## Partie 1 — Définition des outils

```python
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2
```

## Partie 2 — Création des sous-agents

```python
subagent_1 = create_agent(model=model, tools=[square_root])
subagent_2 = create_agent(model=model, tools=[square])
```

## Partie 3 — Créer l'agent principal

Chaque sous-agent est enveloppé dans un `@tool` qui l'invoque et renvoie sa réponse :

```python
@tool
def call_subagent_1(x: float) -> str:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to "
                  "calculate the square root or square of a number.",
)
```

## Partie 4 — Appeler les agents et afficher le résultat

```bash
uv run --active python multi_agents.py
```

### Résultats obtenus

```
Q: What is the square root of 456?
R: The square root of 456 is approximately 21.3542.

Q: What is the square of 12?
R: Thank you for the call! I've got the result. The square of 12 is indeed 144.
```

**Constat :** pour la première question, le `main_agent` appelle `call_subagent_1`, qui invoque `subagent_1` (équipé de `square_root`) ; pour la seconde, il route vers `call_subagent_2` / `subagent_2` (équipé de `square`). Le `main_agent` ne calcule jamais lui-même — il **délègue** systématiquement aux sous-agents spécialisés et reformule leur réponse.

## Partie 5 — Visualiser les 3 agents dans LangGraph Studio

`langgraph.json` expose les 3 graphes (`main_agent`, `subagent_1`, `subagent_2`) :

```json
{
    "graphs": {
        "agent_simple": "./agent_simple.py:agent",
        "main_agent": "./multi_agents.py:main_agent",
        "subagent_1": "./multi_agents.py:subagent_1",
        "subagent_2": "./multi_agents.py:subagent_2"
    },
    "env": "./.env",
    "source": { "kind": "uv", "root": "." }
}
```

```bash
uv run --active langgraph dev
```

Dans Studio, le sélecteur de graphe (en haut) permet de basculer entre `agent_simple`, `main_agent`, `subagent_1` et `subagent_2`, de visualiser chaque graphe (`__start__ → model ⇄ tools → __end__`) et de suivre la trace complète d'une délégation : `main_agent → tools (call_subagent_1) → réponse`.
