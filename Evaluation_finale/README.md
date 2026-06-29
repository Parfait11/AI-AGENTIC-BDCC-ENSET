# Agentic RAG — Gestion des Epidemies de Cholera

Systeme RAG agentique construit avec LangGraph (sans `create_agent`), specialise
dans la surveillance epidemiologique, la reponse de terrain, la rehydratation
orale (SRO) et la prevention du cholera.

## Stack 

- LLM : Ollama local `llama3.2:3b`
- Embeddings : HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Vectorstore : Chroma (persiste dans `data/chroma_db/`)
- Orchestration : LangGraph (`StateGraph`)

## Installation

```bash
uv sync --group dev
```

Assurez-vous qu'Ollama est lance et que le modele est disponible :

```bash
ollama pull llama3.2:3b
```

## Base documentaire

4 guides publics sur le cholera (GTFCC, OMS, MSF) :

| Fichier | Source |
|---|---|
| `who_cholera_outbreak_response.pdf` | GTFCC - Cholera Outbreak Response Field Manual (2024) |
| `who_cholera_surveillance.pdf` | GTFCC - Public Health Surveillance for Cholera (2024) |
| `who_ors_guidelines.pdf` | OMS - Oral Rehydration Salts Guidelines |
| `msf_management_of_a_cholera_epidemic.pdf` | MSF - Management of a Cholera Epidemic |

```bash
uv run python download_pdfs.py   # telecharge les PDF dans data/pdfs/
uv run python check_pdfs.py   #Verifie l'integrite de tous les PDF dans data/pdfs/.
uv run python ingest.py          # construit le vectorstore Chroma
```

## Utilisation
 
```bash
uv run python main.py
```

Chat interactif avec memoire conversationnelle (un `thread_id` par session).

## Architecture du graphe

```bash
uv run python generate_graph.py
```

Genere `graph.mmd` (et `graph.png` si internet disponible). Le graphe suit le
pattern Agentic RAG retrieve -> grade -> generate/rewrite :

- `agent` : le LLM decide d'appeler un outil ou de repondre directement.
- `tools` : execute les outils demandes.
- `grade_documents` : evalue la pertinence des documents recuperes.
- `rewrite_query` : reformule la question si les documents ne sont pas
  pertinents (jusqu'a 2 fois), puis retourne a `agent`.

## Outils

- `retrieve_documents(query)` : recherche semantique dans la base documentaire.
- `compute_ors_dosage(weight_kg, dehydration_level)` : calcule le volume de SRO
  recommande selon le poids du patient et le niveau de deshydratation
  (`'legere'`, `'moderee'`, `'severe'`).
- `estimate_attack_rate(cases, population)` : calcule le taux d'attaque d'une
  epidemie et indique si le seuil d'alerte OMS (> 1 %) est depasse.

## Tests

```bash
uv run pytest -v
```

## Evaluation

```bash
uv run python -m evaluation.run_evaluation
```

Execute 10 questions simples + 10 questions complexes sur le cholera, mesure
le temps de reponse et enregistre les sources dans
`evaluation/results/results.csv`.