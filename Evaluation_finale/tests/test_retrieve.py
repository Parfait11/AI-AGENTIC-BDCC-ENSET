import pytest

from src.config import CHROMA_DIR
from src.tools import retrieve_documents

pytestmark = pytest.mark.skipif(
    not CHROMA_DIR.exists(), reason="Vectorstore non construit (lancez ingest.py)"
)


def test_retrieve_documents_returns_sourced_passages():
    result = retrieve_documents.invoke({"query": "Qu'est-ce qu'un budget personnel ?"})
    assert "Source:" in result
    assert len(result) > 0
