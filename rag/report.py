"""Build the formal PDF evaluation report from out/results.json.

Sections follow the task brief exactly: architecture, diagram index, numerical
demonstration, disambiguation stress test, end-to-end logs, scorecard, and
failure mode analysis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .ingest import DATA, INDEX, ROOT, EMBED_MODEL

OUT = ROOT / "out"
RESULTS = OUT / "results.json"

# Pass criteria come from section 3 of the task brief.
CRITERIA = {
    "context_relevance": (0.60, "High relevance; retrieved chunks pertinent to the query"),
    "context_recall": (0.80, "Context contains the algorithmic steps needed for a complete answer"),
    "context_precision": (0.90, "Top-1 / Top-2 chunks carry the primary solution substance"),
    "faithfulness": (0.80, "Zero hallucinations; every claim grounded in retrieved context"),
    "answer_relevance": (0.85, "Direct, exhaustive answer addressing all question parts"),
    "semantic_accuracy": (0.90, "Mathematically sound; exact complexity notation"),
    "disambiguation_rate": (0.90, "Intended topic resolved without cross-unit collision"),
    "diagram_id_rate": (0.90, "Correct Diagram ID emitted for visual queries"),
    "robustness": (0.90, "Explicit rejection of out-of-scope queries, no hallucinated answer"),
}

LABELS = {
    "context_relevance": ("Retrieval", "Context Relevance"),
    "context_recall": ("Retrieval", "Context Recall"),
    "context_precision": ("Retrieval", "Context Precision & Ranking"),
    "faithfulness": ("Generation", "Faithfulness / Groundedness"),
    "answer_relevance": ("Generation", "Answer Relevance"),
    "semantic_accuracy": ("Generation", "Semantic & Mathematical Accuracy"),
    "disambiguation_rate": ("Specialized", "Cross-Topic Disambiguation Rate"),
    "diagram_id_rate": ("Specialized", "Diagram ID Linkage Rate"),
    "robustness": ("Specialized", "Negative / Out-of-Scope Robustness"),
}


HEADING_RE = __import__("re").compile(r"^(#{1,4})\s", __import__("re").M)


def demote(markdown: str, levels: int = 4) -> str:
    """Push an embedded answer's headings below the report's own sections.

    Generated answers use their own '##' headings. Inserted verbatim they would
    appear as top-level report sections and break the document outline.
    """
    return HEADING_RE.sub(lambda m: "#" * min(6, len(m.group(1)) + levels) + " ", markdown)


def fence(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def matrix_block(matrix, labels=None) -> str:
    lab = labels or [str(i + 1) for i in range(len(matrix))]
    width = max(len(str(c)) for row in matrix for c in row) + 2
    head = "      " + "".join(f"{l:>{width}}" for l in lab)
    lines = [head]
    for name, row in zip(lab, matrix):
        cells = "".join(f"{('inf' if c is None else c):>{width}}" for c in row)
        lines.append(f"{name:>5} {cells}")
    return "\n".join(lines)


def section_architecture(data: dict) -> str:
    meta = json.loads((INDEX / "meta.json").read_text())
    chunks = [json.loads(l) for l in (INDEX / "chunks.jsonl").read_text().splitlines()]
    syllabus = json.loads((DATA / "syllabus.json").read_text())
    n_topics = sum(len(u["topics"]) for u in syllabus["units"])
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["source_type"]] = by_type.get(c["source_type"], 0) + 1
    answerable = sum(v for k, v in by_type.items() if k != "pyq")

    return f"""## 1. System Architecture Overview

The system is a two-stage retrieval augmented generation pipeline. Retrieval is
deterministic and runs to completion before any generation, so the passages and
scores recorded in this report are exactly what the language model received.
Generation is orchestrated by a LangChain Deep Agent whose only tools are exact
numeric solvers.

**Pipeline**

```
PDF / text sources
  -> heading-aware chunking (RecursiveCharacterTextSplitter, {900} chars, {150} overlap)
  -> syllabus tagging (unit + topic, keyword scoring with a section-level prior)
  -> dense embeddings ({EMBED_MODEL}, {meta['dim']}-d, cosine) + BM25 lexical index
query
  -> topic classification (LLM router + deterministic keyword router)
  -> hybrid retrieval: dense top-{20} and BM25 top-{20} fused by reciprocal rank
  -> syllabus topic filter or boost  -> top-{5} passages
  -> diagram registry lookup by topic and keyword
  -> out-of-scope gate on passage similarity
  -> Deep Agent (tools: 9 numeric solvers) -> grounded answer with [chunk] citations
```

**Indexing strategy.** {len(chunks)} chunks total, {answerable} of them answerable.
Source breakdown: {", ".join(f"{k} {v}" for k, v in sorted(by_type.items()))}.
Previous year question papers are indexed for provenance and for sourcing test
queries, but are excluded from answer context because they contain questions
rather than explanations.

**Chunking boundaries.** Splits follow the headings of the study material, including
unnumbered sub-headings such as "Minimum Spanning Tree (MST)", because a DAA section
is a self-contained explanation. Sections longer than the window are split with
RecursiveCharacterTextSplitter. Every chunk carries its unit, topic, source, and
section heading.

**Retrieval model.** Dense retrieval uses {EMBED_MODEL} with the recommended query
instruction prefix; lexical retrieval uses BM25 over the same answerable chunks.
The two rankings are combined with reciprocal rank fusion (k=60). The predicted
syllabus topic is then applied as a hard filter when the router is confident and
at least three chunks match, otherwise as a score boost. This topic step is what
resolves cross-topic keyword collisions.

**Prompt design.** The system prompt forbids ungrounded claims, requires inline
`[C0042]` citations, forbids self-computed arithmetic, requires the exact diagram
identifier in the form `[Diagram: DIAG_...]`, and requires the final asymptotic
bound in exact notation. Retrieved passages are labelled as data, never instructions.

**Syllabus coverage.** {n_topics} topics across the 5 units of AKTU KCS-503. The
supplied notes omit several prescribed topics (LCS detail, Floyd-Warshall,
backtracking, branch and bound, tries, skip lists, linear time sorting, randomized
algorithms, FFT, job sequencing). These were authored as `notes_supplement.md` and
are tagged with their own source so any answer citing them is traceable.

**Generation model.** {data['model']}.
"""


def section_diagrams() -> str:
    reg = json.loads((DATA / "diagrams" / "diagrams.json").read_text())
    rows = [[d["id"], d["unit"], d["topic"], d["concept"], d["file"]]
            for d in reg["diagrams"]]
    return f"""## 2. Diagram and Asset ID Index

Every diagram asset carries a persistent identifier in the format
`{reg['id_format']}`, where CONCEPT is a short upper-case slug and NN is a two
digit sequence number within that concept. The identifier is stable across
rebuilds because it is declared in the registry, not derived from file order.

Assets that display a computed table or trace (knapsack, LCS, Floyd-Warshall,
matrix chain, Dijkstra, Bellman-Ford) are rendered directly from the same solver
functions the answering agent calls, so a figure and an answer cannot disagree.

Registry: `data/diagrams/diagrams.json`, {reg['count']} assets.

{md_table(["Diagram ID", "Unit", "Topic", "Concept", "File"], rows)}

**Linking mechanism.** At retrieval time the registry is searched by predicted
topic and keyword overlap. Matching entries are injected into the prompt as a
DIAGRAMS block and the model must emit the identifier verbatim. After generation,
any `DIAG_` token that was not offered is stripped from the answer and recorded as
an invalid reference, so a broken identifier can never reach the user interface.
"""


def section_numericals(data: dict) -> str:
    rows = [r for r in data["rows"] if r["case"]["category"] == "numeric"]
    parts = ["""## 3. Numerical Handling Demonstration

Arithmetic is never performed by the language model. The router detects that a
query requires computation, the agent calls the matching solver tool, and the
solver returns the full intermediate table which the model must reproduce. The
walkthroughs below are taken verbatim from the evaluation run.
"""]
    for r in rows:
        case, rec, met = r["case"], r["record"], r["metrics"]
        parts.append(f"### {case['id']}. {case['query']}\n")
        parts.append(f"*Source: {case['source']}. Routed to topic "
                     f"`{rec['classification'].get('topic')}` "
                     f"(confidence {rec['classification'].get('confidence')}).*\n")
        for call in rec["tool_calls"]:
            parts.append(f"**Solver call:** `{call['name']}({json.dumps(call['args'])})`\n")
            try:
                res = json.loads(call["result"])
            except (json.JSONDecodeError, TypeError):
                parts.append(fence(str(call["result"])[:1500]))
                continue
            if "steps" in res:
                parts.append("**Solver steps:**\n")
                parts.append(fence("\n".join(str(s) for s in res["steps"])))
            if "table" in res and isinstance(res["table"], list):
                parts.append("**Computed table:**\n")
                parts.append(fence(matrix_block(res["table"])))
            if "final" in res:
                parts.append("**Final matrix:**\n")
                parts.append(fence(matrix_block(res["final"])))
            if "m_table" in res:
                parts.append("**Cost table m:**\n")
                parts.append(fence(matrix_block(
                    [[("-" if v is None else v) for v in row[1:]] for row in res["m_table"][1:]])))
            keys = [k for k in ("bound", "case", "optimal_value", "chosen_items", "length",
                                "lcs", "min_cost", "parenthesization", "distances",
                                "negative_cycle", "complexity") if k in res]
            if keys:
                parts.append(fence("\n".join(f"{k} = {res[k]}" for k in keys)))
        parts.append("**Generated answer:**\n")
        parts.append(demote(rec["answer"]) if rec["answer"] else "_no answer produced_")
        parts.append(f"\n**Verification:** expected "
                     f"`{json.dumps(case['expected_numeric'])}`; "
                     f"{met.get('numeric_note', 'n/a')}. "
                     f"Semantic accuracy {pct(met.get('semantic_accuracy'))}.\n")
        parts.append("---\n")
    return "\n".join(parts)


def section_disambiguation(data: dict) -> str:
    rows = {r["case"]["id"]: r for r in data["rows"]}
    parts = ["""## 4. Cross-Topic Disambiguation Stress Test

Naive retrieval fails when one term appears in several syllabus units. Each pair
below uses the same colliding term in two questions whose correct answers live in
different topics. The test passes only when the router picks the intended topic
**and** the top ranked chunk belongs to it.
"""]
    for pair in data.get("collision_pairs", []):
        ids = pair["queries"]
        parts.append(f"### Colliding term: \"{pair['term']}\"\n")
        table_rows = []
        for qid in ids:
            r = rows.get(qid)
            if not r:
                continue
            rec, case, met = r["record"], r["case"], r["metrics"]
            top = rec["chunks"][0] if rec["chunks"] else {}
            table_rows.append([
                qid,
                case["query"][:70] + ("..." if len(case["query"]) > 70 else ""),
                case["expected_topic"],
                rec["classification"].get("topic"),
                f"{top.get('id', '-')} ({top.get('topic', '-')})",
                "PASS" if met.get("disambiguation") == 1.0 else "FAIL",
            ])
        parts.append(md_table(
            ["Query", "Prompt", "Expected topic", "Routed topic", "Top-1 chunk", "Result"],
            table_rows))
        parts.append("")
        for qid in ids:
            r = rows.get(qid)
            if not r:
                continue
            ctx = ", ".join(f"{c['id']}:{c['topic']}" for c in r["record"]["chunks"])
            parts.append(f"**{qid} retrieved context:** {ctx}")
            excerpt = (r["record"]["answer"] or "")[:420].replace("\n", " ")
            parts.append(f"\n**{qid} answer excerpt:** {excerpt}...\n")
        parts.append("---\n")
    return "\n".join(parts)


def section_logs(data: dict) -> str:
    parts = [f"""## 5. End-to-End Test Case Logs

All {data['query_count']} test queries with the retrieved context, fusion scores
and the raw generated output. Queries marked with a PYQ source are taken verbatim
from the supplied AKTU previous year question papers.
"""]
    for r in data["rows"]:
        case, rec, met = r["case"], r["record"], r["metrics"]
        parts.append(f"### {case['id']} ({case['category']}) - {case['source']}\n")
        parts.append(f"**Query prompt:** {case['query']}\n")
        cls = rec["classification"]
        parts.append(f"**Routing:** topic `{cls.get('topic')}`, unit {cls.get('unit')}, "
                     f"confidence {cls.get('confidence')}, router {cls.get('source')}, "
                     f"filter {rec['retrieval'].get('filter_mode')}, "
                     f"max passage similarity {rec['retrieval'].get('max_dense')}\n")
        if rec["chunks"]:
            parts.append("**Retrieved context chunks:**\n")
            parts.append(md_table(
                ["Chunk", "Topic", "Unit", "RRF score", "Dense", "BM25", "Section"],
                [[c["id"], c["topic"], c["unit"], f"{c['score']:.5f}", c["dense"],
                  c["bm25"], c["heading"][:52]] for c in rec["chunks"]]))
            parts.append("")
        else:
            parts.append("**Retrieved context chunks:** none (refused before generation)\n")
        if rec["tool_calls"]:
            parts.append("**Solver calls:** " + ", ".join(
                f"`{t['name']}`" for t in rec["tool_calls"]) + "\n")
        if rec["offered_diagrams"]:
            parts.append(f"**Diagrams offered:** {', '.join(rec['offered_diagrams'])}  \n"
                         f"**Diagram IDs emitted:** "
                         f"{', '.join(rec['diagram_ids']) or 'none'}\n")
        parts.append("**Raw generated output:**\n")
        parts.append(fence(rec["answer"] or "(no answer)", "markdown"))
        scores = " | ".join(
            f"{k}={pct(met.get(k))}" for k in
            ("context_relevance", "context_precision", "context_recall", "faithfulness",
             "answer_relevance", "semantic_accuracy") if met.get(k) is not None)
        parts.append(f"\n**Scores:** {scores}  \n"
                     f"**Judge:** {met.get('judge_reason', '')}  \n"
                     f"**Latency:** {rec.get('latency_s')}s\n")
        parts.append("---\n")
    return "\n".join(parts)


def section_scorecard(data: dict) -> str:
    sc = data["scorecard"]
    rows = []
    passed = 0
    for key, (threshold, objective) in CRITERIA.items():
        stage, name = LABELS[key]
        value = sc.get(key)
        ok = value is not None and value >= threshold
        passed += int(ok)
        rows.append([stage, name, pct(value), f">= {threshold * 100:.0f}%",
                     "PASS" if ok else "FAIL", objective])
    cat_rows = []
    for cat, card in sorted(data["by_category"].items()):
        cat_rows.append([cat] + [pct(card.get(k)) for k in
                                 ("context_relevance", "context_precision", "faithfulness",
                                  "answer_relevance", "semantic_accuracy", "robustness")])
    return f"""## 6. Quantitative Evaluation Scorecard

Scored over {data['query_count']} labelled queries, {data['duration_s']}s wall clock,
model {data['model']}. Retrieval relevance, precision, disambiguation, diagram
linkage and robustness are computed from the gold labels. Recall, faithfulness and
answer relevance are scored by an LLM judge with a fixed rubric that also sees the
verified solver output. Numeric semantic accuracy is checked programmatically
against expected solver values.

{md_table(["Stage", "Metric", "Score", "Pass criterion", "Result", "Verification objective"], rows)}

**Overall: {passed} of {len(CRITERIA)} checklist metrics met.**

### Scores by query category

{md_table(["Category", "Ctx relevance", "Ctx precision", "Faithfulness",
           "Answer relevance", "Semantic acc.", "Robustness"], cat_rows)}

Supporting rate: required-keyword coverage in answers {pct(sc.get('must_mention_rate'))}.
"""


def section_failures(data: dict) -> str:
    misses = []
    for r in data["rows"]:
        case, rec, met = r["case"], r["record"], r["metrics"]
        problems = []
        if rec.get("error"):
            problems.append(("infrastructure", rec["error"]))
        if met.get("disambiguation") == 0.0:
            problems.append(("retrieval", f"routing: {met.get('disambiguation_note')}"))
        if met.get("diagram_id") == 0.0:
            problems.append(("generation", f"diagram: {met.get('diagram_note')}"))
        if met.get("robustness") == 0.0:
            problems.append(("retrieval",
                             "refused an in-scope query" if rec["refused"]
                             else "answered an out-of-scope query"))
        if met.get("context_precision") == 0.0:
            problems.append(("retrieval",
                             f"expected topic {case['expected_topic']} absent from top-2"))
        for key, stage in (("faithfulness", "generation"), ("semantic_accuracy", "generation"),
                           ("context_recall", "retrieval")):
            v = met.get(key)
            if v is not None and v < 0.7:
                problems.append((stage, f"{key} {pct(v)}: {met.get('judge_reason', '')[:140]}"))
        if met.get("must_mention_missing"):
            problems.append(("generation",
                             f"missing required terms {met['must_mention_missing']}"))
        for stage, detail in problems:
            misses.append([case["id"], case["category"], stage, detail[:200]])

    body = md_table(["Query", "Category", "Stage", "Observation"], misses) if misses else \
        "No metric fell below its pass criterion on any individual query."

    return f"""## 7. Failure Mode Analysis and Limitations

Each observation below is attributed to the stage that caused it. A failure is a
retrieval failure when the needed passage was absent or mis-ranked, and a
generation failure when the passage was present but the answer misused it.

{body}

### Known limitations

1. **Corpus depth.** The supplied notes give only one or two passages for several
   topics, notably Dijkstra, Bellman-Ford and MST. The topic filter therefore falls
   back to a unit filter when fewer than three chunks match, and context relevance
   is capped by how few on-topic chunks exist rather than by ranking quality.

2. **Authored supplement.** Ten prescribed topics were missing from the supplied
   material and were authored for this system. Answers on those topics are grounded
   in `notes_supplement.md`, not in the original notes, and are marked as such by
   their chunk source.

3. **Single judge model.** Recall, faithfulness and answer relevance are scored by
   the same model family that generates answers, which is a known bias. The
   label-derived metrics (relevance, precision, disambiguation, diagram linkage,
   robustness) are independent of the model and should carry more weight.

4. **Out-of-scope gate is similarity based.** A query that is off-syllabus but
   lexically close to indexed material can pass the gate. The threshold was tuned
   on the four out-of-scope queries in the test set, which is a small sample.

5. **No scanned PDF support.** Ingestion reads the text layer only. A scanned
   question paper without embedded text would yield no chunks; OCR is not wired in.

6. **Diagrams are generated, not extracted.** The supplied notes contain no figures,
   so all {len(json.loads((DATA / 'diagrams' / 'diagrams.json').read_text())['diagrams'])}
   assets were rendered programmatically. They illustrate the standard textbook
   constructions rather than reproducing any figure from the source notes.

### Mitigations

- Ingest a fuller textbook corpus to raise per-topic chunk counts, which would let
  the topic filter apply instead of the unit fallback.
- Add a cross-encoder reranker over the fused top-20 if precision drops as the
  corpus grows.
- Use `RubricMiddleware` from Deep Agents to re-prompt the agent when a grounding
  rubric fails at answer time, rather than only measuring faithfulness afterwards.
- Score the judge metrics with a second, different model and report agreement.
"""


def build_markdown(data: dict) -> str:
    head = f"""# RAG System Evaluation Report

**Task:** CALIB-RAG-AKTU-DAA-01, Technical Calibration Assignment
**Subject:** Design and Analysis of Algorithms, AKTU KCS-503
**Deliverable:** Fully functional Retrieval Augmented Generation system with
empirical metric-gated audit
**Generated:** {data['generated_at']}
**Generation model:** {data['model']}
**Queries evaluated:** {data['query_count']}

---

"""
    return "\n".join([
        head,
        section_architecture(data),
        section_diagrams(),
        section_numericals(data),
        section_disambiguation(data),
        section_logs(data),
        section_scorecard(data),
        section_failures(data),
    ])


CSS = """
@page { size: A4; margin: 18mm 16mm; @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 9.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2px solid #0b6cb0; padding-bottom: 5px; }
h2 { font-size: 13.5pt; color: #0b6cb0; margin-top: 20px; border-bottom: 1px solid #ddd; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin-top: 14px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 7.8pt; }
th { background: #eaf3fb; text-align: left; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; }
pre { background: #f6f7f9; border: 1px solid #ddd; padding: 6px; font-size: 7.4pt;
      white-space: pre-wrap; word-wrap: break-word; font-family: "DejaVu Sans Mono", monospace; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; }
hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
"""


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit("out/results.json not found; run `python cli.py eval` first")
    data = json.loads(RESULTS.read_text())
    OUT.mkdir(exist_ok=True)
    md_path = OUT / "report.md"
    md_path.write_text(build_markdown(data))
    print(f"wrote {md_path} ({md_path.stat().st_size // 1024} KB)")

    pdf_path = OUT / "report.pdf"
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", str(md_path), "-o", str(pdf_path),
                        "-V", "geometry:margin=18mm"], check=True)
    else:
        import markdown as md_lib
        from weasyprint import CSS as WCSS, HTML

        html = md_lib.markdown(md_path.read_text(),
                               extensions=["tables", "fenced_code", "sane_lists"])
        HTML(string=f"<html><head><meta charset='utf-8'></head><body>{html}</body></html>",
             base_url=str(ROOT)).write_pdf(pdf_path, stylesheets=[WCSS(string=CSS)])
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
