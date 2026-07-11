# LAB 6 : Contexte et État dans un agent

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Comprendre et mettre en pratique la différence entre **contexte** (données immuables fournies à l'invocation) et **état** (données mutables qui évoluent et persistent pendant la conversation) dans un agent LangChain/LangGraph.

---

## Structure du projet

```
Lab6-Contexte_et_Etat/
├── agent_context.py   # Partie 1 : le contexte (immuable, par invocation)
├── agent_state.py     # Partie 2 : l'état (mutable, persisté par thread)
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

## Concept : Contexte vs État

```
CONTEXTE                              ÉTAT
────────────────────                  ────────────────────
Immuable pendant l'invocation         Mutable pendant la conversation
Fourni via invoke(context=...)        Modifié par les tools via Command
Pas persisté entre les appels         Persisté par thread_id (checkpointer)
Ex : profil utilisateur, paramètres   Ex : préférences révélées au fil du chat
```

---

## Partie 1 — Le contexte (`agent_context.py`)

Le contexte représente des informations connues **avant** la conversation (profil du lecteur d'une bibliothèque : nom, niveau d'abonnement, langue préférée). Il est déclaré via `context_schema` et fourni à chaque appel via `invoke(..., context=...)`.

```python
@dataclass
class ReaderProfile:
    name: str = "Khalid"
    membership: str = "standard"
    preferred_language: str = "français"

@tool
def get_membership_level(runtime: ToolRuntime) -> str:
    """Get the library membership level of the current reader."""
    return runtime.context.membership

agent_with_context = create_agent(
    model=model,
    tools=[get_membership_level, get_preferred_language],
    context_schema=ReaderProfile,
)

response = agent_with_context.invoke(
    {"messages": [HumanMessage(content="Quel est mon niveau d'abonnement ?")]},
    context=ReaderProfile(),
)
```

### Exécution

```bash
uv run --active python agent_context.py
```

### Résultats obtenus

```
--- Partie 2 : Agent sans accès au contexte ---
Je ne suis pas en mesure de vous fournir des informations sur votre compte
d'abonnement à une bibliothèque spécifique. [...] Contactez la bibliothèque
directement [...]

--- Partie 3 : Agent avec accès au contexte ---
Votre niveau d'abonnement à la bibliothèque est standard.

--- Partie 4 : Changement de contexte (lecteur premium) ---
Votre niveau d'abonnement à la bibliothèque est de niveau "premium".
```

**Constat :** sans tool pour lire `runtime.context`, le LLM ne peut pas accéder au contexte (Partie 2). Avec un tool dédié, il y accède directement (Partie 3). Le contexte change librement d'un appel à l'autre — chaque `invoke` peut représenter un lecteur différent (Partie 4).

---

## Partie 2 — L'état (`agent_state.py`)

L'état représente des informations **découvertes pendant** la conversation et qui doivent persister entre les tours d'échange (le genre littéraire favori du lecteur, révélé en cours de discussion). On étend `AgentState`, et un tool le modifie via `Command(update=...)`.

```python
class LibraryState(AgentState):
    favourite_genre: str

@tool
def remember_favourite_genre(genre: str, runtime: ToolRuntime) -> Command:
    """Store the reader's favourite literary genre once they reveal it."""
    return Command(update={
        "favourite_genre": genre,
        "messages": [ToolMessage(f"Genre favori enregistré : {genre}",
                                 tool_call_id=runtime.tool_call_id)]
    })

@tool
def recall_favourite_genre(runtime: ToolRuntime) -> str:
    """Read the reader's favourite literary genre from the persisted state."""
    try:
        return runtime.state["favourite_genre"]
    except KeyError:
        return "Aucun genre favori enregistré pour le moment."

agent = create_agent(
    model=model,
    tools=[remember_favourite_genre, recall_favourite_genre],
    state_schema=LibraryState,
    checkpointer=InMemorySaver(),
)
```

### Exécution

```bash
uv run --active python agent_state.py
```

### Résultats obtenus

```
--- Partie 6 : enregistrer le genre favori ---
Vous aimez les romans de science-fiction ! [...] je peux vous suggérer
quelques titres populaires du genre [...]
État courant : science-fiction

--- Partie 6 (suite) : récupérer le genre persisté ---
Votre genre littéraire favori est bien la science-fiction ! [...]

--- Partie 7 : nouveau thread, état vide ---
Je suis désolé, mais je n'ai pas d'informations sur vos préférences
littéraires. Si vous voulez partager votre genre favori avec moi [...]
```

**Constat :** dans le même `thread_id` (`reader-1`), l'agent se souvient du genre favori d'un tour à l'autre grâce au `checkpointer` (Partie 6). Avec un nouveau `thread_id` (`reader-2`), l'état repart de zéro — la persistance est isolée par thread (Partie 7).

---

## Comparaison

| | Contexte (`agent_context.py`) | État (`agent_state.py`) |
|---|---|---|
| **Mutabilité** | Immuable pendant l'invocation | Mutable, modifié par les tools |
| **Origine** | Fourni par l'appelant (`context=`) | Construit pendant la conversation |
| **Persistance** | Aucune — redéfini à chaque appel | Persisté par `thread_id` via `checkpointer` |
| **Mécanisme de lecture** | `runtime.context` | `runtime.state[...]` |
| **Mécanisme d'écriture** | — (immuable) | `Command(update={...})` |
| **Cas d'usage typique** | Profil utilisateur, paramètres de session | Préférences révélées, mémoire conversationnelle |
