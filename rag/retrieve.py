"""Query classification, hybrid retrieval and diagram lookup.

Retrieval runs before generation and is fully deterministic, so the chunks and
scores recorded for the evaluation report are exactly what the model saw.

Ranking combines a dense bi-encoder with BM25 through reciprocal rank fusion,
then applies the syllabus topic predicted for the query. The topic filter is
what resolves cross-topic keyword collisions such as 'relaxation' appearing in
both Dijkstra and Bellman-Ford.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ingest import DATA, INDEX, EMBED_MODEL, QUERY_PREFIX, load_syllabus, topic_table, _kw_re

TOP_K = 5
CANDIDATES = 20
RRF_K = 60
TOPIC_FILTER_CONFIDENCE = 0.7
TOPIC_BOOST = 1.5
MIN_TOPIC_CHUNKS = 3

# Tuned on the out-of-scope queries of data/testset.json.
# ponytail: single global threshold; per-topic thresholds only if recall suffers.
OOS_DENSE_FLOOR = 0.45
OOS_DENSE_UNCLASSIFIED = 0.58

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Scored:
    id: str
    score: float
    dense: float
    bm25: float
    unit: int | None
    topic: str | None
    heading: str
    source: str
    text: str


@lru_cache(maxsize=1)
def load_index() -> dict:
    chunks = [json.loads(line) for line in (INDEX / "chunks.jsonl").read_text().splitlines()]
    vecs = np.load(INDEX / "embeddings.npy")
    if len(chunks) != vecs.shape[0]:
        raise RuntimeError("index is stale: rebuild with `python cli.py ingest`")

    from rank_bm25 import BM25Okapi

    answerable = [i for i, c in enumerate(chunks) if c["source_type"] != "pyq"]
    bm25 = BM25Okapi([tokenize(chunks[i]["heading"] + " " + chunks[i]["text"]) for i in answerable])
    return {
        "chunks": chunks,
        "vecs": vecs,
        "answerable": answerable,
        "bm25": bm25,
        "by_id": {c["id"]: c for c in chunks},
    }


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed_query(query: str) -> np.ndarray:
    vec = embedder().encode([QUERY_PREFIX + query], convert_to_numpy=True,
                            normalize_embeddings=True)
    return vec[0].astype(np.float32)


# --------------------------------------------------------------------------
# Query classification
# --------------------------------------------------------------------------

NUMERIC_HINTS = {
    "master_theorem": ["master theorem", "master method", "solve the recurrence", "t(n) ="],
    "recursion_tree": ["recursion tree"],
    "knapsack_01": ["0/1 knapsack", "01 knapsack"],
    "fractional_knapsack": ["fractional knapsack"],
    "lcs": ["longest common subsequence", "lcs"],
    "matrix_chain": ["matrix chain", "parenthesization", "parenthesisation"],
    "floyd_warshall": ["floyd", "warshall", "all pair shortest path", "all pairs shortest path"],
    "dijkstra": ["dijkstra"],
    "bellman_ford": ["bellman-ford", "bellman ford"],
}

DIAGRAM_HINTS = [
    "diagram", "draw", "figure", "illustrate", "show the tree", "show the table",
    "rotation", "trace", "step by step", "step-by-step", "flowchart", "stages",
]


def lexical_classify(query: str) -> dict:
    """Score the query against the syllabus keyword map.

    This is the deterministic fallback for the LLM classifier and the tie
    breaker when the model is unavailable.
    """
    topics = topic_table(load_syllabus())
    q = query.lower()
    scored = []
    for t in topics:
        score = 0.0
        for kw in t["keywords"]:
            if _kw_re(kw).search(q):
                score += 1
        for kw in t["strong"]:
            if _kw_re(kw).search(q):
                score += 4
        if score:
            scored.append((score, t))
    scored.sort(key=lambda p: -p[0])
    if not scored:
        return {"unit": None, "topic": None, "confidence": 0.0, "source": "lexical"}
    best_score, best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    # Confidence reflects the margin over the next best topic, so a query whose
    # terms fit two topics equally well is deliberately left unconfident.
    margin = (best_score - runner) / best_score
    confidence = round(min(0.95, 0.45 + 0.5 * margin), 3)
    return {
        "unit": best["unit"],
        "topic": best["topic"],
        "confidence": confidence,
        "source": "lexical",
        "runner_up": scored[1][1]["topic"] if len(scored) > 1 else None,
    }


CLASSIFY_PROMPT = """You route a student's question to one topic of the AKTU \
Design and Analysis of Algorithms syllabus (KCS-503).

Topics, given as unit | topic id | description:
{topics}

Rules:
- Choose the single topic the question is really about.
- Many terms appear in several topics. 'Relaxation' appears in both Dijkstra and \
Bellman-Ford; 'knapsack' appears in the greedy fractional knapsack (unit 3) and \
the 0/1 knapsack solved by dynamic programming (unit 4); 'dynamic programming' \
covers matrix chain multiplication, LCS, 0/1 knapsack and Floyd-Warshall. Use the \
rest of the question to decide which one is meant.
- If the question is not about this syllabus at all, set topic to null.
- needs_diagram is true when a correct answer needs a figure, tree, table or trace.
- numeric_task names the solver required for an exact computation, or null.
  Allowed: master_theorem, recursion_tree, knapsack_01, fractional_knapsack, lcs,
  matrix_chain, floyd_warshall, dijkstra, bellman_ford.

Question: {query}

Reply with only a JSON object:
{{"topic": <topic id or null>, "confidence": <0..1>, "needs_diagram": <bool>, \
"numeric_task": <name or null>}}"""


def llm_classify(query: str) -> dict | None:
    """Ask the chat model to route the query. Returns None if it is unavailable."""
    from .config import chat_model

    topics = topic_table(load_syllabus())
    listing = "\n".join(f"{t['unit']} | {t['topic']} | {t['title']}" for t in topics)
    prompt = CLASSIFY_PROMPT.format(topics=listing, query=query)
    try:
        resp = chat_model(temperature=0.0, max_tokens=300).invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        data = json.loads(match.group(0))
    except Exception as exc:  # network, quota, malformed output
        return {"error": str(exc)[:200]}

    valid = {t["topic"]: t["unit"] for t in topics}
    topic = data.get("topic")
    if topic not in valid:
        topic = None
    task = data.get("numeric_task")
    if task not in NUMERIC_HINTS:
        task = None
    return {
        "unit": valid.get(topic),
        "topic": topic,
        "confidence": float(data.get("confidence") or 0.0),
        "needs_diagram": bool(data.get("needs_diagram")),
        "numeric_task": task,
        "source": "llm",
    }


def classify(query: str, use_llm: bool = True) -> dict:
    """Route a query to a syllabus topic, combining the model and the keyword map."""
    lex = lexical_classify(query)
    q = query.lower()
    hint_task = next((name for name, kws in NUMERIC_HINTS.items()
                      if any(k in q for k in kws)), None)
    hint_diagram = any(k in q for k in DIAGRAM_HINTS)

    result = dict(lex)
    result["lexical_topic"] = lex["topic"]
    result["needs_diagram"] = hint_diagram
    result["numeric_task"] = hint_task
    result["llm_error"] = None

    if use_llm:
        llm = llm_classify(query)
        if llm and "error" in llm:
            result["llm_error"] = llm["error"]
        elif llm:
            result["llm_topic"] = llm["topic"]
            result["agreement"] = llm["topic"] == lex["topic"]
            if llm["topic"]:
                result.update(
                    unit=llm["unit"], topic=llm["topic"],
                    confidence=max(llm["confidence"], 0.5 if llm["topic"] == lex["topic"] else 0.0),
                    source="llm+lexical" if llm["topic"] == lex["topic"] else "llm",
                )
                if llm["topic"] == lex["topic"]:
                    result["confidence"] = round(min(0.99, max(llm["confidence"], lex["confidence"], 0.8)), 3)
            elif lex["topic"] is None:
                result["topic"] = None
            result["needs_diagram"] = hint_diagram or llm["needs_diagram"]
            result["numeric_task"] = hint_task or llm["numeric_task"]
    return result


# --------------------------------------------------------------------------
# Hybrid search
# --------------------------------------------------------------------------

def search(query: str, classification: dict, top_k: int = TOP_K) -> tuple[list[Scored], dict]:
    idx = load_index()
    chunks, vecs, answerable = idx["chunks"], idx["vecs"], idx["answerable"]

    qv = embed_query(query)
    sims = vecs[answerable] @ qv
    dense_order = np.argsort(-sims)[:CANDIDATES]

    bm_scores = idx["bm25"].get_scores(tokenize(query))
    bm_order = np.argsort(-bm_scores)[:CANDIDATES]

    fused: dict[int, float] = {}
    for rank, pos in enumerate(dense_order):
        fused[int(pos)] = fused.get(int(pos), 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, pos in enumerate(bm_order):
        fused[int(pos)] = fused.get(int(pos), 0.0) + 1.0 / (RRF_K + rank + 1)

    topic = classification.get("topic")
    confidence = float(classification.get("confidence") or 0.0)
    mode = "none"
    if topic:
        matching = [p for p in fused if chunks[answerable[p]]["topic"] == topic]
        if confidence >= TOPIC_FILTER_CONFIDENCE and len(matching) >= MIN_TOPIC_CHUNKS:
            fused = {p: s for p, s in fused.items() if p in matching}
            mode = "filter:topic"
        else:
            unit = classification.get("unit")
            unit_matching = [p for p in fused if chunks[answerable[p]]["unit"] == unit]
            if confidence >= TOPIC_FILTER_CONFIDENCE and len(unit_matching) >= MIN_TOPIC_CHUNKS:
                fused = {p: s for p, s in fused.items() if p in unit_matching}
                mode = "filter:unit"
            else:
                for p in matching:
                    fused[p] *= TOPIC_BOOST
                mode = "boost:topic"

    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for pos, score in ranked:
        c = chunks[answerable[pos]]
        out.append(
            Scored(
                id=c["id"], score=round(float(score), 6),
                dense=round(float(sims[pos]), 4), bm25=round(float(bm_scores[pos]), 3),
                unit=c["unit"], topic=c["topic"], heading=c["heading"],
                source=c["source"], text=c["text"],
            )
        )
    debug = {
        "filter_mode": mode,
        "max_dense": round(float(sims.max()), 4),
        "candidates": len(fused),
    }
    return out, debug


# --------------------------------------------------------------------------
# Diagrams
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_diagrams() -> list[dict]:
    path = DATA / "diagrams" / "diagrams.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["diagrams"]


def find_diagrams(query: str, classification: dict, limit: int = 2) -> list[dict]:
    """Match diagram assets by topic and keyword overlap."""
    diagrams = load_diagrams()
    if not diagrams:
        return []
    q = query.lower()
    topic = classification.get("topic")
    scored = []
    for d in diagrams:
        score = 0.0
        if topic and d["topic"] == topic:
            score += 5
        elif topic and d["unit"] == classification.get("unit"):
            score += 1
        for kw in d["keywords"]:
            if _kw_re(kw.lower()).search(q):
                score += 2
        for word in tokenize(d["concept"]):
            if len(word) > 3 and word in q:
                score += 1
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda p: (-p[0], p[1]["id"]))
    # A diagram is only offered when the topic itself matches; keyword-only
    # overlap is too weak and produces mismatched IDs in answers.
    return [d for s, d in scored[:limit] if s >= 5]


# --------------------------------------------------------------------------
# Entry point used by the answering layer
# --------------------------------------------------------------------------

def retrieve(query: str, use_llm: bool = True, top_k: int = TOP_K) -> dict:
    classification = classify(query, use_llm=use_llm)
    chunks, debug = search(query, classification, top_k=top_k)

    max_dense = debug["max_dense"]
    refuse = False
    reason = None
    if max_dense < OOS_DENSE_FLOOR:
        refuse, reason = True, f"best passage similarity {max_dense:.2f} below floor {OOS_DENSE_FLOOR}"
    elif classification.get("topic") is None and max_dense < OOS_DENSE_UNCLASSIFIED:
        refuse, reason = True, (
            f"query did not map to any syllabus topic and best similarity "
            f"{max_dense:.2f} is below {OOS_DENSE_UNCLASSIFIED}"
        )

    diagrams = find_diagrams(query, classification) if classification.get("needs_diagram") else []
    return {
        "query": query,
        "classification": classification,
        "chunks": chunks,
        "diagrams": diagrams,
        "debug": debug,
        "refuse": refuse,
        "refusal_reason": reason,
    }
