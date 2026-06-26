## Prérequis

- Python >= 3.10
- [Ollama](https://ollama.com/) avec `llama3.2:3b` (`ollama pull llama3.2:3b`)
- Dépendances installées :

```bash
pip install langchain langchain-community langchain-ollama langgraph \
            sentence-transformers pypdf sqlalchemy


PDF → Chargement → Segmentation → Embeddings → VectorStore → Recherche sémantique → Agent


User question
     │
     ▼
  Agent LLM (llama3.2:3b)
     │
     └── tool: search_handbook(query)
                    │
                    ▼
           InMemoryVectorStore
           similarity_search()
                    │
                    ▼
           chunk le plus proche
           (acmecorp-employee-handbook.pdf)
                    │
                    ▼
           Réponse contextualisée



Question NL → Agent LLM → sql_query tool → SQLite → Résultat → Réponse NL


--- Test du tool sql_query ---
[(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), ...]

--- Réponse de l'Agent SQL ---
Here are the first 5 artists in the database:
1. AC/DC
2. Accept
3. Aerosmith
4. Alanis Morissette
5. Alice In Chains


Table Artist:
  - ArtistId  INTEGER (PK)
  - Name      TEXT