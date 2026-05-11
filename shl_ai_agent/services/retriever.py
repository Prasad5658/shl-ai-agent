import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = None
index = None
metadata = None


def _load_resources():
    global model, index, metadata

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if index is None:
        index = faiss.read_index("embeddings/faiss_index.bin")

    if metadata is None:
        with open("embeddings/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "hiring",
    "in",
    "need",
    "of",
    "or",
    "role",
    "skills",
    "test",
    "tests",
    "the",
    "to",
    "with",
}

SYNONYMS = {
    "developer": {"developer", "programmer", "software", "coding", "programming"},
    "java": {"java", "j2ee", "spring"},
    "manager": {"manager", "management", "leadership", "supervisor"},
    "stakeholder": {"stakeholder", "communication", "collaboration", "interpersonal"},
    "communication": {"communication", "verbal", "written", "interpersonal"},
    "personality": {"personality", "behavior", "behaviour", "opq", "motivational"},
    "cognitive": {"cognitive", "ability", "aptitude", "reasoning"},
    "sales": {"sales", "selling", "customer"},
}


def _tokens(text):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def _expanded_tokens(text):
    tokens = _tokens(text)
    expanded = set(tokens)

    for token in tokens:
        expanded.update(SYNONYMS.get(token, set()))

    return expanded


def _document_text(item):
    return " ".join(
        str(value)
        for value in (
            item.get("name", ""),
            item.get("description", ""),
            item.get("test_type", ""),
            item.get("duration", ""),
            item.get("remote", ""),
            item.get("adaptive", ""),
        )
    )


def _keyword_score(query_terms, item):
    doc_terms = _tokens(_document_text(item))
    name_terms = _tokens(item.get("name", ""))
    type_terms = _tokens(item.get("test_type", ""))

    if not query_terms:
        return 0.0

    overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
    name_boost = 0.35 * len(query_terms & name_terms)
    type_boost = 0.2 * len(query_terms & type_terms)

    return overlap + name_boost + type_boost


def _semantic_candidates(query, candidate_k):
    _load_resources()
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, candidate_k)

    candidates = {}

    for distance, idx in zip(distances[0], indices[0]):
        item = dict(metadata[idx])
        item["_semantic_score"] = 1.0 / (1.0 + float(distance))
        candidates[idx] = item

    return candidates


def retrieve_assessments(query, top_k=5, candidate_k=40):
    _load_resources()
    query_terms = _expanded_tokens(query)
    candidates = _semantic_candidates(query, min(candidate_k, len(metadata)))

    for idx, item in enumerate(metadata):
        score = _keyword_score(query_terms, item)
        if score > 0 and idx not in candidates:
            candidate = dict(item)
            candidate["_semantic_score"] = 0.0
            candidates[idx] = candidate

    ranked = []

    for item in candidates.values():
        keyword = _keyword_score(query_terms, item)
        semantic = item.get("_semantic_score", 0.0)
        total = semantic + keyword

        if "java" in query_terms and "java" in item.get("name", "").lower():
            total += 1.0

        if query_terms & {"personality", "behavior", "behaviour", "opq"}:
            if "personality" in item.get("test_type", "").lower() or "opq" in item.get("name", "").lower():
                total += 1.0

        item["_score"] = round(total, 4)
        ranked.append(item)

    ranked.sort(key=lambda item: item["_score"], reverse=True)

    return ranked[:top_k]


def find_assessments_by_name(query, top_k=3):
    _load_resources()
    query_terms = _expanded_tokens(query)
    ranked = []

    for item in metadata:
        candidate = dict(item)
        score = _keyword_score(query_terms, candidate)

        if query.lower() in candidate.get("name", "").lower():
            score += 2.0

        candidate["_score"] = round(score, 4)
        ranked.append(candidate)

    ranked.sort(key=lambda item: item["_score"], reverse=True)
    return ranked[:top_k]
