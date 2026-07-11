# LAB 4 : Le Model Context Protocol (MCP)

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Intégrer le **Model Context Protocol (MCP)** avec LangChain pour connecter des agents LLM à des serveurs MCP locaux et distants via différents transports (stdio, HTTP streaming).

---

## Structure du projet

```
Lab4-MCP/
├── mcp_local_server.py      # Serveur MCP local (stdio) — tools + resources + prompts
├── mcp_http_server.py       # Serveur MCP HTTP local (streamable-http)
├── agentMCP.py              # Partie 1 : Agent avec serveur MCP local
├── agentMCPTime.py          # Partie 2 : Agent avec serveur MCP de temps
├── agentMCPDistant.py       # Partie 3 : Agent avec serveur MCP distant (HTTP)
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Prérequis

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) installé
- [Ollama](https://ollama.com/) avec le modèle `llama3.2:3b` (`ollama pull llama3.2:3b`)
- Clé API Tavily (`TAVILY_API_KEY`) dans le fichier `.env`

### Installation

```bash
# Copier le fichier d'environnement
cp .env.example .env
# Remplir les clés API dans .env

# Installer les dépendances
uv sync

# Sur Windows : installer pywin32 manuellement (contournement antivirus)
python -m ensurepip
pip3 install pywin32
```

---

## Parties

### Partie 1 — Serveur MCP local (stdio)

Le serveur `mcp_local_server.py` expose :
- **Tool** : `search_web` — recherche web via Tavily
- **Resource** : README du dépôt `langchain-mcp-adapters` (GitHub)
- **Prompt** : système de l'assistant LangChain

```bash
uv run --active python agentMCP.py
```

**Résultat :** L'agent récupère dynamiquement les tools/resources/prompts du serveur MCP, effectue une recherche web et répond à la question.

```
{'messages': [..., AIMessage(content='The langchain-mcp-adapters library is a
lightweight wrapper that makes Anthropic Model Context Protocol (MCP) tools
compatible with LangChain and LangGraph...')]}
```

---

### Partie 2 — Serveur MCP de temps (time server)

Utilise le package `mcp-server-time` pour exposer l'heure en temps réel par timezone.

```bash
uv run --active python agentMCPTime.py
```

**Résultat :**
```
The current time in Japan is 3:30 AM on Saturday (June 6th, 2026),
considering the 13-hour time difference from New York.
```

> **Note Windows :** Le serveur `mcp-server-time` doit être installé dans le `.venv` via `pip3 install mcp-server-time` (au lieu de `uvx`) en raison du problème de cache `pywin32` sur Windows.

---

### Partie 3 — Serveur MCP distant (HTTP streaming)

Démonstration d'une connexion à un serveur MCP via le transport `streamable-http`. Un serveur de voyage local est démarré automatiquement dans un thread séparé.

```bash
uv run --active python agentMCPDistant.py
```

**Résultat :**
```
Serveur MCP HTTP démarré sur http://127.0.0.1:8000/mcp
Tools disponibles : ['search_flights', 'get_flight_price']

Here are the direct flight options from Rabat to Agadir on August 31st:
* Atlas Blue (AT) - Flight 601: Departing at 08:00, arriving at 09:15
* Atlas Blue (AT) - Flight 603: Departing at 14:30, arriving at 15:45
* Atlas Blue (AT) - Flight 605: Departing at 19:00, arriving at 20:15
```

---

## Architecture MCP

```
Agent LLM (LangChain)
    │
    └── MultiServerMCPClient
            │
            ├── stdio transport ──────► mcp_local_server.py (python subprocess)
            ├── stdio transport ──────► mcp-server-time (executable)
            └── streamable-http ──────► http://127.0.0.1:8000/mcp (uvicorn)
```

---

## Dépendances principales

| Package | Rôle |
|---|---|
| `langchain-mcp-adapters` | Client MCP pour LangChain/LangGraph |
| `mcp` | Implémentation du protocole MCP |
| `fastmcp` | Framework serveur MCP simplifié |
| `mcp-server-time` | Serveur MCP de temps (timezone) |
| `langchain-ollama` | Modèle LLM local via Ollama |
| `tavily-python` | Recherche web |
| `langgraph` | Runtime d'agent |
