"""Run the labelled test set and score the nine checklist metrics.

Retrieval-stage metrics that can be decided from the gold labels are computed
programmatically, so they do not depend on a model's opinion. The three
judgement-based metrics use a single LLM-as-judge call per query with a fixed
rubric, following the LangSmith RAG evaluation pattern.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import ask
from .config import chat_model, model_label
from .ingest import DATA, ROOT

RESULTS = ROOT / "out" / "results.json"
# The endpoint returns empty completions when hit too hard; 2 workers is the
# rate at which every judge call came back with content.
WORKERS = 3

JUDGE_PROMPT = """You are grading one answer produced by a retrieval augmented \
generation system for the AKTU subject Design and Analysis of Algorithms.

You are given the QUESTION, the CONTEXT the system retrieved, and the \
ANSWER it generated. The CONTEXT may include VERIFIED SOLVER OUTPUT blocks: these \
are exact values computed by trusted code, and any answer content matching them is \
fully grounded. Grade strictly and independently on four criteria, each a \
number from 0.0 to 1.0.

context_recall: does the CONTEXT contain every fact, step and formula needed to \
answer the QUESTION completely? Grade the context, not the answer. 1.0 means \
nothing needed is missing.

faithfulness: is every claim, step and formula in the ANSWER supported by the \
CONTEXT or by a tool result quoted in the answer? Any unsupported or invented \
claim caps this below 0.5. Correct textbook knowledge that is absent from the \
CONTEXT still counts as unsupported.

answer_relevance: does the ANSWER directly and completely address every part of \
the QUESTION, without evasion or padding? 1.0 means all parts answered.

semantic_accuracy: is the technical content correct? Check recurrence solutions, \
arithmetic, table values, pseudocode logic and asymptotic notation. 1.0 means \
everything is mathematically sound and the complexity notation is exact.

QUESTION
{query}

CONTEXT
{context}

ANSWER
{answer}

Reply with only a JSON object and no other text:
{{"context_recall": <0..1>, "faithfulness": <0..1>, "answer_relevance": <0..1>, \
"semantic_accuracy": <0..1>, "reason": "<one sentence>"}}"""


def load_testset() -> dict:
    return json.loads((DATA / "testset.json").read_text())


JUDGE_CONTEXT_LIMIT = 9000
JUDGE_ANSWER_LIMIT = 6000


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[... truncated for grading ...]"


def judge(query: str, context: str, answer: str) -> dict:
    """Grade one answer. Retries once, because a judge that returns no JSON would
    otherwise silently void three of the nine metrics."""
    prompt = JUDGE_PROMPT.format(
        query=query,
        context=_clip(context, JUDGE_CONTEXT_LIMIT),
        answer=_clip(answer, JUDGE_ANSWER_LIMIT),
    )
    last = ""
    for attempt in range(3):
        if attempt:
            time.sleep(2 * attempt)
        try:
            resp = chat_model(temperature=0.0, max_tokens=1500,
                                      reasoning_effort="low").invoke(prompt)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            last = text
            match = re.search(r"\{.*\}", text, re.S)
            if match is None:
                continue
            data = json.loads(match.group(0))
            return {
            "context_recall": float(data.get("context_recall", 0.0)),
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "answer_relevance": float(data.get("answer_relevance", 0.0)),
            "semantic_accuracy": float(data.get("semantic_accuracy", 0.0)),
                "reason": str(data.get("reason", ""))[:300],
                "error": None,
            }
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
    return {"context_recall": None, "faithfulness": None, "answer_relevance": None,
            "semantic_accuracy": None, "reason": "",
            "error": f"judge returned no JSON after 3 attempts; last response: {last[:180]!r}"}


def _numeric_match(expected: dict, tool_calls: list[dict], answer: str) -> tuple[float, str]:
    """Check the solver was called and that the answer reproduces its values."""
    if not tool_calls:
        return 0.0, "no solver tool was called"
    results = []
    for call in tool_calls:
        try:
            results.append(json.loads(call["result"]) if call["result"] else {})
        except (json.JSONDecodeError, TypeError):
            results.append({})
    ok, notes = [], []
    for key, want in expected.items():
        got = next((r.get(key) for r in results if key in r), None)
        if got is None:
            ok.append(False)
            notes.append(f"{key} absent from tool output")
            continue
        same = got == want
        if not same and isinstance(want, float):
            same = abs(float(got) - want) < 1e-6
        ok.append(bool(same))
        if not same:
            notes.append(f"{key}: tool returned {got!r}, expected {want!r}")
    # The answer must also carry the number, not just the tool.
    for key, want in expected.items():
        if isinstance(want, (int, float)) and str(want) not in answer:
            notes.append(f"{key} value {want} not reproduced in the answer text")
            ok.append(False)
    score = sum(ok) / len(ok) if ok else 0.0
    return score, "; ".join(notes) or "tool output matches the expected values"


def score_query(case: dict, record: dict) -> dict:
    """Compute every metric that applies to one query."""
    cat = case["category"]
    chunks = record["chunks"]
    topics = [c["topic"] for c in chunks]
    expected_topic = case["expected_topic"]
    # A comparison question ("differentiate backtracking and branch and bound")
    # is pertinent to more than one topic, so relevance is scored against a set.
    acceptable = set(case.get("acceptable_topics") or ([expected_topic] if expected_topic else []))
    answer = record["answer"]
    m: dict[str, float | None] = {}

    # --- Retrieval stage, decided from the gold labels -------------------
    if cat == "oos":
        m["context_relevance"] = None
        m["context_precision"] = None
    else:
        m["context_relevance"] = (
            round(sum(1 for t in topics if t in acceptable) / len(topics), 3)
            if topics else 0.0
        )
        m["context_precision"] = 1.0 if any(t in acceptable for t in topics[:2]) else 0.0

    # --- Specialised domain metrics --------------------------------------
    classified = record["classification"].get("topic")
    if cat == "disambiguation":
        hit = classified == expected_topic and topics[:1] == [expected_topic]
        forbidden = case.get("must_not_mention") or []
        leaked = [f for f in forbidden if f.lower() in answer.lower()]
        m["disambiguation"] = 1.0 if (hit and not leaked) else 0.0
        m["disambiguation_note"] = (
            f"classified {classified}, top-1 {topics[0] if topics else None}"
            + (f", leaked {leaked}" if leaked else "")
        )
    else:
        m["disambiguation"] = None

    if cat == "diagram":
        want = set(case["expected_diagram_ids"])
        got = set(record["diagram_ids"])
        invalid = record.get("invalid_diagram_ids") or []
        m["diagram_id"] = 1.0 if (want == got and not invalid) else 0.0
        m["diagram_note"] = f"expected {sorted(want)}, emitted {sorted(got)}" + (
            f", invalid {invalid}" if invalid else "")
    else:
        # Emitting a diagram id that was never offered is a failure anywhere.
        m["diagram_id"] = 0.0 if record.get("invalid_diagram_ids") else None

    if cat == "oos":
        m["robustness"] = 1.0 if record["refused"] else 0.0
    else:
        m["robustness"] = 0.0 if record["refused"] else 1.0

    # --- Generation stage -------------------------------------------------
    if cat == "oos":
        # A refusal needs no judge: the label is the whole test.
        m.update(context_recall=None, faithfulness=None, answer_relevance=None,
                 semantic_accuracy=None, judge_reason="out of scope, graded on refusal only")
        return m

    mentioned = [k for k in (case.get("must_mention") or []) if k.lower() in answer.lower()]
    m["must_mention_rate"] = (
        round(len(mentioned) / len(case["must_mention"]), 3) if case.get("must_mention") else None
    )
    m["must_mention_missing"] = [
        k for k in (case.get("must_mention") or []) if k.lower() not in answer.lower()
    ]

    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    if record["tool_calls"]:
        # Verified solver output is grounding evidence just as much as a passage.
        tools = "\n\n".join(
            f"[VERIFIED SOLVER OUTPUT: {t['name']}({json.dumps(t['args'])[:300]})]\n"
            f"{(t['result'] or '')[:2000]}"
            for t in record["tool_calls"]
        )
        context = f"{context}\n\n{tools}"
    verdict = judge(case["query"], context, answer) if answer else {
        "context_recall": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0,
        "semantic_accuracy": 0.0, "reason": "no answer produced", "error": None}
    m["context_recall"] = verdict["context_recall"]
    m["faithfulness"] = verdict["faithfulness"]
    m["answer_relevance"] = verdict["answer_relevance"]
    m["judge_reason"] = verdict["reason"]
    m["judge_error"] = verdict["error"]

    if case.get("expected_numeric"):
        score, note = _numeric_match(case["expected_numeric"], record["tool_calls"], answer)
        m["semantic_accuracy"] = score
        m["numeric_note"] = note
        m["numeric_tool_ok"] = 1.0 if any(
            t["name"] == case["expected_numeric_task"] for t in record["tool_calls"]) else 0.0
    else:
        m["semantic_accuracy"] = verdict["semantic_accuracy"]

    return m


def run_case(case: dict) -> dict:
    record = ask(case["query"])
    metrics = score_query(case, record)
    return {"case": case, "record": record, "metrics": metrics}


def scorecard(rows: list[dict]) -> dict:
    def mean(key: str, categories: set[str] | None = None) -> float | None:
        vals = [
            r["metrics"][key] for r in rows
            if r["metrics"].get(key) is not None
            and (categories is None or r["case"]["category"] in categories)
        ]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "context_relevance": mean("context_relevance"),
        "context_recall": mean("context_recall"),
        "context_precision": mean("context_precision"),
        "faithfulness": mean("faithfulness"),
        "answer_relevance": mean("answer_relevance"),
        "semantic_accuracy": mean("semantic_accuracy"),
        "disambiguation_rate": mean("disambiguation", {"disambiguation"}),
        "diagram_id_rate": mean("diagram_id", {"diagram"}),
        "robustness": mean("robustness"),
        "must_mention_rate": mean("must_mention_rate"),
    }


def by_category(rows: list[dict]) -> dict:
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["case"]["category"], []).append(r)
    return {c: scorecard(rs) for c, rs in cats.items()}


def main(only: str | None = None) -> None:
    ts = load_testset()
    cases = ts["queries"]
    if only:
        wanted = {c.strip().upper() for c in only.split(",")}
        cases = [c for c in cases if c["id"] in wanted or c["category"] == only]
    print(f"running {len(cases)} queries against {model_label()} with {WORKERS} workers")

    started = time.time()
    evaluated_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row in pool.map(run_case, cases):
            evaluated_rows.append(row)
            met = row["metrics"]
            flag = "ok "
            if row["record"].get("error"):
                flag = "ERR"
            elif met.get("disambiguation") == 0.0 or met.get("diagram_id") == 0.0 \
                    or met.get("robustness") == 0.0:
                flag = "MISS"
            print(f"  {flag:4s} {row['case']['id']} {row['case']['category']:14s} "
                  f"{row['case']['query'][:58]}")

    if only and RESULTS.exists():
        try:
            existing = json.loads(RESULTS.read_text())
            existing_rows = {r["case"]["id"]: r for r in existing.get("rows", [])}
        except Exception:
            existing_rows = {}
        for r in evaluated_rows:
            existing_rows[r["case"]["id"]] = r
        rows = list(existing_rows.values())
    else:
        rows = evaluated_rows

    rows.sort(key=lambda r: r["case"]["id"])
    out = {
        "model": model_label(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - started, 1),
        "query_count": len(rows),
        "scorecard": scorecard(rows),
        "by_category": by_category(rows),
        "collision_pairs": ts.get("collision_pairs", []),
        "rows": rows,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS} in {out['duration_s']}s")
    print(json.dumps(out["scorecard"], indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
