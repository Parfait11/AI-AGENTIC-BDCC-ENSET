from pathlib import Path

from langchain.tools import tool

from src.config import RETRIEVAL_K
from src.vectorstore import get_vectorstore


@tool
def compute_ors_dosage(weight_kg: float, dehydration_level: str) -> dict:
    """Calcule la dose de solution de rehydratation orale (SRO) recommandee
    selon le poids du patient et le niveau de deshydratation.

    Args:
        weight_kg: poids du patient en kilogrammes.
        dehydration_level: niveau de deshydratation — 'legere', 'moderee' ou 'severe'.
    """
    level = dehydration_level.strip().lower()

    # Volumes de reference OMS (ml/kg sur 4h)
    ml_per_kg = {"legere": 50, "moderee": 75, "severe": 100}

    if level not in ml_per_kg:
        return {
            "error": (
                f"Niveau de deshydratation inconnu : '{dehydration_level}'. "
                "Valeurs acceptees : 'legere', 'moderee', 'severe'."
            )
        }

    volume_ml = weight_kg * ml_per_kg[level]
    return {
        "weight_kg": weight_kg,
        "dehydration_level": level,
        "ors_volume_ml_4h": round(volume_ml, 0),
        "ors_volume_liters_4h": round(volume_ml / 1000, 2),
        "note": (
            "Volume a administrer sur 4 heures. "
            "Réévaluer apres chaque heure. "
            "En cas de vomissements, fractionner en petites gorgees frequentes."
        ),
    }


@tool
def estimate_attack_rate(cases: int, population: int) -> dict:
    """Calcule le taux d'attaque (TA) d'une épidémie de cholera dans une zone donnee.

    Args:
        cases: nombre de cas de cholera confirmes ou suspects.
        population: population totale de la zone concernee.
    """
    if population <= 0:
        return {"error": "La population doit etre superieure a zero."}

    attack_rate_percent = (cases / population) * 100
    cases_per_1000 = (cases / population) * 1000

    # Seuil d'alerte OMS : taux d'attaque > 1% en situation d'urgence
    alert = attack_rate_percent > 1.0

    return {
        "cases": cases,
        "population": population,
        "attack_rate_percent": round(attack_rate_percent, 4),
        "cases_per_1000": round(cases_per_1000, 2),
        "who_alert_threshold_exceeded": alert,
        "note": (
            "Seuil d'alerte OMS : taux d'attaque > 1 % en contexte d'urgence. "
            "Un taux eleve necessite une reponse immediate."
        ),
    }


@tool
def retrieve_documents(query: str) -> str:
    """Recherche dans la base documentaire sur le cholera (surveillance,
    reponse de terrain, rehydratation orale, prevention et controle) les
    passages les plus pertinents pour une question.

    Args:
        query: la question ou les mots-cles a rechercher.
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    if not docs:
        return "Aucun document pertinent trouve."
    formatted = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "inconnu")).name
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


TOOLS = [retrieve_documents, compute_ors_dosage, estimate_attack_rate]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}