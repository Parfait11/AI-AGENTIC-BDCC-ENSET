"""Construit et persiste le vectorstore Chroma a partir des PDF de data/pdfs/.

A lancer une seule fois (ou apres ajout/modification de documents) :
    uv run python ingest.py
"""
from src.vectorstore import build_vectorstore

 
def main() -> None:
    vectorstore = build_vectorstore()
    count = vectorstore._collection.count()
    print(f"Vectorstore construit et persiste dans data/chroma_db/ : {count} chunks indexes.")


if __name__ == "__main__":
    main()
