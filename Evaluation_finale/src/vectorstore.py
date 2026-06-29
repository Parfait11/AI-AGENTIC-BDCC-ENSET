from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_DIR,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Retourne le modèle d'embedding avec détection automatique du device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Utilisation du device: {device}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 4}
    )


def is_valid_pdf(path: Path) -> bool:
    """Verifie que le fichier commence par l'en-tete PDF (%PDF)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def load_single_pdf(pdf_path: Path) -> list:
    """Charge un seul PDF (utilisé pour le parallélisme)."""
    try:
        if not is_valid_pdf(pdf_path):
            print(f"[IGNORE] PDF invalide : {pdf_path.name}")
            return []
        docs = PyPDFLoader(str(pdf_path)).load()
        print(f"[OK] {pdf_path.name} — {len(docs)} pages chargees")
        return docs
    except Exception as exc:
        print(f"[ERREUR] Impossible de lire {pdf_path.name} : {exc}")
        return []


def build_vectorstore() -> Chroma:
    """Charge les PDF valides de data/pdfs/, les decoupe en chunks et construit
    le vectorstore Chroma persiste dans data/chroma_db/."""
    
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouve dans {PDF_DIR}. Lancez d'abord download_pdfs.py."
        )

    # Chargement parallèle des PDF
    documents = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_single_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
        for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Chargement PDF"):
            docs = future.result()
            documents.extend(docs)

    if not documents:
        raise ValueError(
            "Aucun document valide n'a pu etre charge. "
            "Verifiez les PDF dans data/pdfs/ (lancez check_pdfs.py pour diagnostiquer)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"[OK] {len(chunks)} chunks crees a partir de {len(documents)} pages")

    embeddings = get_embeddings()
    
    print("Construction du vectorstore (cela peut prendre quelques minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    
    print(f"[OK] Vectorstore construit avec {vectorstore._collection.count()} chunks")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Charge le vectorstore Chroma deja persiste (construit par ingest.py)."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            f"Vectorstore introuvable dans {CHROMA_DIR}. Lancez d'abord : uv run python ingest.py"
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm

# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.config import (
#     CHROMA_DIR,
#     CHUNK_OVERLAP,
#     CHUNK_SIZE,
#     COLLECTION_NAME,
#     EMBEDDING_MODEL_NAME,
#     PDF_DIR,
# )

# # Option 1: Utiliser un modèle plus petit et plus rapide
# # Décommentez si vous voulez forcer un modèle rapide
# # EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dims, 2x plus rapide

# def get_embeddings() -> HuggingFaceEmbeddings:
#     return HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         model_kwargs={"device": "cuda"},  # ou "mps" pour Mac, "cpu" sinon
#         encode_kwargs={"batch_size": 32}  # Traiter plusieurs textes à la fois
#     )


# def is_valid_pdf(path: Path) -> bool:
#     """Verifie que le fichier commence par l'en-tete PDF (%PDF)."""
#     try:
#         with open(path, "rb") as f:
#             return f.read(4) == b"%PDF"
#     except Exception:
#         return False


# def load_single_pdf(pdf_path: Path) -> list:
#     """Charge un seul PDF (utilisé pour le parallélisme)."""
#     try:
#         if not is_valid_pdf(pdf_path):
#             print(f"[IGNORE] PDF invalide : {pdf_path.name}")
#             return []
#         docs = PyPDFLoader(str(pdf_path)).load()
#         print(f"[OK] {pdf_path.name} — {len(docs)} pages chargees")
#         return docs
#     except Exception as exc:
#         print(f"[ERREUR] Impossible de lire {pdf_path.name} : {exc}")
#         return []


# def build_vectorstore() -> Chroma:
#     """Charge les PDF valides de data/pdfs/, les decoupe en chunks et construit
#     le vectorstore Chroma persiste dans data/chroma_db/."""
    
#     pdf_files = sorted(PDF_DIR.glob("*.pdf"))

#     if not pdf_files:
#         raise FileNotFoundError(
#             f"Aucun PDF trouve dans {PDF_DIR}. Lancez d'abord download_pdfs.py."
#         )

#     # OPTIMISATION 1: Chargement parallèle des PDF (2-4x plus rapide)
#     documents = []
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = {executor.submit(load_single_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
#         for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Chargement PDF"):
#             docs = future.result()
#             documents.extend(docs)

#     if not documents:
#         raise ValueError(
#             "Aucun document valide n'a pu etre charge. "
#             "Verifiez les PDF dans data/pdfs/ (lancez check_pdfs.py pour diagnostiquer)."
#         )

#     # OPTIMISATION 2: Découpage plus efficace (chunks plus petits = moins d'embeddings)
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,  # Essayez 500 pour plus de rapidité
#         chunk_overlap=CHUNK_OVERLAP,  # Essayez 50 pour plus de rapidité
#         add_start_index=True
#     )
#     chunks = splitter.split_documents(documents)
#     print(f"[OK] {len(chunks)} chunks crees a partir de {len(documents)} pages")

#     # OPTIMISATION 3: Création du vectorstore avec batch_size
#     embeddings = get_embeddings()
    
#     print("Construction du vectorstore (cela peut prendre quelques minutes)...")
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         collection_name=COLLECTION_NAME,
#         persist_directory=str(CHROMA_DIR),
#     )
    
#     print(f"[OK] Vectorstore construit avec {vectorstore._collection.count()} chunks")
#     return vectorstore


# def get_vectorstore() -> Chroma:
#     """Charge le vectorstore Chroma deja persiste (construit par ingest.py)."""
#     if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
#         raise FileNotFoundError(
#             f"Vectorstore introuvable dans {CHROMA_DIR}. Lancez d'abord : uv run python ingest.py"
#         )
#     return Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=get_embeddings(),
#         persist_directory=str(CHROMA_DIR),
#     )


# # from pathlib import Path

# # from langchain_chroma import Chroma
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_text_splitters import RecursiveCharacterTextSplitter

# # from src.config import (
# #     CHROMA_DIR,
# #     CHUNK_OVERLAP,
# #     CHUNK_SIZE,
# #     COLLECTION_NAME,
# #     EMBEDDING_MODEL_NAME,
# #     PDF_DIR,
# # )


# # def get_embeddings() -> HuggingFaceEmbeddings:
# #     return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# # def is_valid_pdf(path: Path) -> bool:
# #     """Verifie que le fichier commence par l'en-tete PDF (%PDF)."""
# #     try:
# #         with open(path, "rb") as f:
# #             return f.read(4) == b"%PDF"
# #     except Exception:
# #         return False


# # def build_vectorstore() -> Chroma:
# #     """Charge les PDF valides de data/pdfs/, les decoupe en chunks et construit
# #     le vectorstore Chroma persiste dans data/chroma_db/."""
# #     documents = []
# #     pdf_files = sorted(PDF_DIR.glob("*.pdf"))

# #     if not pdf_files:
# #         raise FileNotFoundError(
# #             f"Aucun PDF trouve dans {PDF_DIR}. Lancez d'abord download_pdfs.py."
# #         )

# #     for pdf_path in pdf_files:
# #         if not is_valid_pdf(pdf_path):
# #             print(f"[IGNORE] PDF invalide (HTML ou corrompu) : {pdf_path.name}")
# #             continue
# #         try:
# #             docs = PyPDFLoader(str(pdf_path)).load()
# #             documents.extend(docs)
# #             print(f"[OK] {pdf_path.name} — {len(docs)} pages chargees")
# #         except Exception as exc:
# #             print(f"[ERREUR] Impossible de lire {pdf_path.name} : {exc}")

# #     if not documents:
# #         raise ValueError(
# #             "Aucun document valide n'a pu etre charge. "
# #             "Verifiez les PDF dans data/pdfs/ (lancez check_pdfs.py pour diagnostiquer)."
# #         )

# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
# #     )
# #     chunks = splitter.split_documents(documents)
# #     print(f"[OK] {len(chunks)} chunks crees a partir de {len(documents)} pages")

# #     return Chroma.from_documents(
# #         documents=chunks,
# #         embedding=get_embeddings(),
# #         collection_name=COLLECTION_NAME,
# #         persist_directory=str(CHROMA_DIR),
# #     )


# # def get_vectorstore() -> Chroma:
# #     """Charge le vectorstore Chroma deja persiste (construit par ingest.py)."""
# #     if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
# #         raise FileNotFoundError(
# #             f"Vectorstore introuvable dans {CHROMA_DIR}. Lancez d'abord : uv run python ingest.py"
# #         )
# #     return Chroma(
# #         collection_name=COLLECTION_NAME,
# #         embedding_function=get_embeddings(),
# #         persist_directory=str(CHROMA_DIR),
# #     )