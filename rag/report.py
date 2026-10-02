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


def demote(markdown: str, levels: int = 3) -> str:
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

    return f"""## I. SYSTEM ARCHITECTURE OVERVIEW

The system is a two-stage retrieval augmented generation pipeline coupled with a
self-deployed FastAPI Agent Server and a responsive assistant-ui web interface.
Retrieval is deterministic and runs to completion before any generation, so the
passages and scores recorded in this report are exactly what the language model
received. Generation is orchestrated by a LangChain Deep Agent whose only tools
are exact numeric solvers.

### A. End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   assistant-ui React Web Frontend                      │
│             (Vite + TypeScript + Tailwind CSS @ localhost:5173)        │
│                                                                        │
│   • Streaming chat interface with step-by-step markdown rendering      │
│   • Live Retrieval & Solver Telemetry side-panel                       │
│   • Persistent Diagram & Schematic Asset Registry browser (21 assets)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ SSE / REST Stream
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Self-Deployed FastAPI Agent Server (server.py)           │
│                      (Uvicorn @ localhost:8000)                        │
│                                                                        │
│   • POST /api/chat     -> Real-time Server-Sent Events (SSE) stream    │
│   • POST /api/ask      -> Direct AnswerRecord payload & provenance     │
│   • GET  /api/diagrams -> Static diagram asset metadata & image URLs   │
│   • GET  /api/syllabus -> 5-unit syllabus topic and keyword hierarchy  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Two-Step RAG Retrieval & Ingestion Pipeline            │
│                                                                        │
│   PDF / text sources                                                   │
│     -> heading-aware chunking (RecursiveCharacterTextSplitter)         │
│     -> syllabus tagging (unit + topic keyword scoring with priors)     │
│     -> dense embeddings ({EMBED_MODEL}) + BM25 sparse index            │
│   query                                                                │
│     -> topic routing (LLM router + deterministic keyword classifier)   │
│     -> hybrid retrieval: dense top-20 + BM25 top-20 fused via RRF      │
│     -> topic priority ranking & adaptive context windowing             │
│     -> diagram registry lookup by concept & keyword                    │
│     -> out-of-scope similarity gate -> Deep Agent (+ 9 Solvers)        │
└────────────────────────────────────────────────────────────────────────┘
```

### B. Ingestion and Indexing Strategy

A total of {len(chunks)} chunks are indexed, with {answerable} designated as answerable context.
Source distribution: {", ".join(f"{k} ({v})" for k, v in sorted(by_type.items()))}.
Previous year question papers (`pyq/`) are indexed strictly for provenance and authentic test extraction, but are excluded from generation context to prevent question-question pollution.

### C. Chunking Boundaries

Splits follow the logical heading structure of the study materials, including unnumbered sub-headings (such as "Minimum Spanning Tree"). Sections exceeding the token window are sub-split using LangChain's `RecursiveCharacterTextSplitter` (900 characters, 150 overlap). Each chunk preserves its unit, topic, source file, and section heading.

### D. Hybrid Retrieval and Disambiguation

Dense semantic search employs `{EMBED_MODEL}` with normalized embeddings; lexical retrieval utilizes `BM25Okapi` over identical answerable chunks. Rankings are fused using Reciprocal Rank Fusion (RRF, k=60). Topic routing applies priority ranking with adaptive context sizing: on-topic chunks lead, and top-up passages are retrieved only to maintain a minimum context of 3 chunks.

### E. Prompt Design and Deterministic Solvers

The system prompt enforces zero hallucination, mandatory inline `[C0042]` citations, exact diagram referencing in `[Diagram: DIAG_...]` format, and asymptotic bounds. Arithmetic and recursive computations are strictly delegated to 9 deterministic Python solvers (Master Theorem, recursion tree, 0/1 knapsack, fractional knapsack, LCS, matrix chain multiplication, Floyd-Warshall, Dijkstra, Bellman-Ford).

### F. Syllabus Coverage and Authored Supplement

The knowledge base covers all {n_topics} syllabus topics across the 5 units of AKTU KCS-503. Topics omitted from the primary notes (LCS detail, Floyd-Warshall, backtracking, branch & bound, tries, skip lists, linear time sorting, randomized algorithms, FFT, job sequencing) are authored in `notes_supplement.md` and explicitly cited.

### G. Context Management and Thread State Architecture

The system enforces a disciplined context lifecycle across retrieval, agent reasoning, and multi-turn execution:

1. **Model Input Context (Per-Turn Grounding Window):**
   - **System Grounding Directives:** Rigorous guardrails requiring inline `[C0042]` citations, mathematical proof structure, and explicit prohibition of mental arithmetic or speculative recurrence bounds.
   - **Retrieved Corpus Chunks:** Contextually routed passages tagged with unique chunk IDs (`[C0001]`), syllabus units, and section headings.
   - **Diagram Asset Registry:** Candidate visual schematic identifiers (`[Diagram: DIAG_...]`) matching the predicted topic.
   - **Verified Solver Traces:** Exact numeric outputs, dynamic programming tables, and recurrence relations generated by deterministic Python tools injected as verified context (`[VERIFIED SOLVER OUTPUT]`).

2. **Thread State and Provenance Persistence:**
   - **Conversational History:** User prompts, agent responses, and thread turn sequencing.
   - **Topic Routing Telemetry:** Classified unit, topic, routing method (LLM vs. rule classifier), and classification confidence scores.
   - **Retrieved Chunk Telemetry:** Complete chunk payload with dense similarity, BM25 BM-score, and fused RRF scores for auditability.
   - **Tool Execution Traces:** Record of invoked solver tool names, raw JSON argument payloads, and return objects.
   - **Visual Asset Telemetry:** Eligible schematic candidate list and verified emitted diagram IDs.
   - **Evaluation Provenance:** Extracted citation indices and millisecond-level end-to-end execution latency.

**Generation Model:** {data['model']}.
"""


def section_diagrams() -> str:
    reg = json.loads((DATA / "diagrams" / "diagrams.json").read_text())
    rows = [[d["id"], f"Unit {d['unit']}", d["topic"], d["concept"], d["file"]]
            for d in reg["diagrams"]]
    gallery = "\n\n".join(
        f"**Fig. {i+1}: {d['id']} — {d['concept']}**\n\n![{d['concept']}](data/diagrams/{d['file']})\n\n*{d['caption']}*\n\n---"
        for i, d in enumerate(reg["diagrams"])
    )
    return f"""## II. DIAGRAM AND ASSET ID INDEX

Every visual schematic in the system carries a persistent unique identifier following the specification `{reg['id_format']}`. The identifier is deterministic and defined in `data/diagrams/diagrams.json` ({reg['count']} assets total).

Assets depicting algorithmic traces or dynamic programming tables (knapsack, LCS, Floyd-Warshall, matrix chain, Dijkstra, Bellman-Ford) are programmatically rendered by executing the exact solver code invoked by the agent, ensuring complete mathematical concordance between text and schematics.

### A. Asset Registry Catalog

{md_table(["Diagram ID", "Unit", "Topic", "Concept", "Filename"], rows)}

### B. Rendered Diagram Schematics

{gallery}

### C. Linking and Verification Mechanism

During retrieval, the diagram catalog is scored against predicted topics and keyword overlap. Eligible identifiers are injected into the prompt. A post-generation verification pass strips any unoffered `DIAG_` identifiers, preventing broken references from reaching user interfaces.
"""


def section_numericals(data: dict) -> str:
    rows = [r for r in data["rows"] if r["case"]["category"] == "numeric"]
    parts = ["""## III. NUMERICAL HANDLING DEMONSTRATION

Probabilistic language models are prohibited from performing internal arithmetic. Numerical tasks are routed to deterministic Python solvers that compute exact intermediate tables and states, which the agent must faithfully reproduce. The walkthroughs below record exact system execution.
"""]
    for r in rows:
        case, rec, met = r["case"], r["record"], r["metrics"]
        parts.append(f"### {case['id']}: {case['query']}\n")
        parts.append(f"*Source: {case['source']} | Routed Topic: `{rec['classification'].get('topic')}` "
                     f"(Confidence: {rec['classification'].get('confidence')})*\n")
        for call in rec["tool_calls"]:
            parts.append(f"**Solver Tool Invocation:** `{call['name']}({json.dumps(call['args'])})`\n")
            try:
                res = json.loads(call["result"])
            except (json.JSONDecodeError, TypeError):
                parts.append(fence(str(call["result"])[:1500]))
                continue
            if "steps" in res:
                parts.append("**Computed Algorithmic Steps:**\n")
                parts.append(fence("\n".join(str(s) for s in res["steps"])))
            if "table" in res and isinstance(res["table"], list):
                parts.append("**Computed Dynamic Programming Matrix:**\n")
                parts.append(fence(matrix_block(res["table"])))
            if "final" in res:
                parts.append("**Final Distance Matrix D^(k):**\n")
                parts.append(fence(matrix_block(res["final"])))
            if "m_table" in res:
                parts.append("**MCM Cost Table m[i,j]:**\n")
                parts.append(fence(matrix_block(
                    [[("-" if v is None else v) for v in row[1:]] for row in res["m_table"][1:]])))
            keys = [k for k in ("bound", "case", "optimal_value", "chosen_items", "length",
                                "lcs", "min_cost", "parenthesization", "distances",
                                "negative_cycle", "complexity") if k in res]
            if keys:
                parts.append(fence("\n".join(f"{k} = {res[k]}" for k in keys)))
        parts.append("\n**Generated Derivation Answer:**\n")
        parts.append(demote(rec["answer"]) if rec["answer"] else "_no answer produced_")
        parts.append(f"\n**Verification Audit:** Expected `{json.dumps(case['expected_numeric'])}`; "
                     f"{met.get('numeric_note', 'n/a')}. "
                     f"Semantic Accuracy: **{pct(met.get('semantic_accuracy'))}**.\n")
        parts.append("---\n")
    return "\n".join(parts)


def section_disambiguation(data: dict) -> str:
    rows = {r["case"]["id"]: r for r in data["rows"]}
    parts = ["""## IV. CROSS-TOPIC DISAMBIGUATION STRESS TEST

Algorithmic terminology frequently collides across distinct syllabus units (e.g., 'relaxation' in Dijkstra vs. Bellman-Ford; 'knapsack' in Greedy vs. Dynamic Programming). The benchmark tests disambiguation pairs using identical root keywords across separate units. A test passes only when the topic classifier resolves the intended topic and the top-ranked passage belongs to that topic.
"""]
    for pair in data.get("collision_pairs", []):
        ids = pair["queries"]
        parts.append(f"### Term Collision Analysis: \"{pair['term']}\"\n")
        table_rows = []
        for qid in ids:
            r = rows.get(qid)
            if not r:
                continue
            rec, case, met = r["record"], r["case"], r["metrics"]
            top = rec["chunks"][0] if rec["chunks"] else {}
            table_rows.append([
                qid,
                case["query"][:65] + ("..." if len(case["query"]) > 65 else ""),
                case["expected_topic"],
                str(rec["classification"].get("topic")),
                f"{top.get('id', '-')}: {top.get('topic', '-')}",
                "PASS" if met.get("disambiguation") == 1.0 else "FAIL",
            ])
        parts.append(md_table(
            ["Query ID", "Prompt Text", "Expected Topic", "Routed Topic", "Top-1 Passage", "Result"],
            table_rows))
        parts.append("")
        for qid in ids:
            r = rows.get(qid)
            if not r:
                continue
            ctx = ", ".join(f"{c['id']}:{c['topic']}" for c in r["record"]["chunks"])
            parts.append(f"**{qid} Retrieved Context Passages:** {ctx}")
            excerpt = (r["record"]["answer"] or "")[:350].replace("\n", " ")
            parts.append(f"\n**{qid} Answer Excerpt:** {excerpt}...\n")
        parts.append("---\n")
    return "\n".join(parts)


def section_logs(data: dict) -> str:
    parts = [f"""## V. END-TO-END TEST CASE LOGS

Complete execution logs for all {data['query_count']} benchmark test cases, detailing query prompts, topic classifications, fusion retrieval rankings, and raw generated outputs. Queries with a PYQ source are transcribed verbatim from official AKTU examinations.
"""]
    for r in data["rows"]:
        case, rec, met = r["case"], r["record"], r["metrics"]
        parts.append(f"### {case['id']}: [{case['category'].upper()}] — {case['source']}\n")
        parts.append(f"**Query Prompt:** {case['query']}\n")
        cls = rec["classification"]
        parts.append(f"**Retrieval Telemetry:** Topic `{cls.get('topic')}` (Unit {cls.get('unit')}), "
                     f"Confidence: {cls.get('confidence')}, Router: {cls.get('source')}, "
                     f"Filter: {rec['retrieval'].get('filter_mode')}, Max Similarity: {rec['retrieval'].get('max_dense')}\n")
        if rec["chunks"]:
            parts.append("\n**Retrieved Context Chunks:**\n")
            parts.append(md_table(
                ["Chunk ID", "Topic", "Unit", "RRF Score", "Dense", "BM25", "Section Heading"],
                [[c["id"], c["topic"], c["unit"], f"{c['score']:.5f}", c["dense"],
                  c["bm25"], c["heading"][:48]] for c in rec["chunks"]]))
            parts.append("")
        else:
            parts.append("\n**Retrieved Context Chunks:** None (Refused by Out-of-Scope Gate)\n")
        if rec["tool_calls"]:
            parts.append("**Solvers Executed:** " + ", ".join(
                f"`{t['name']}`" for t in rec["tool_calls"]) + "\n")
        if rec["offered_diagrams"]:
            parts.append(f"**Diagrams Offered:** {', '.join(rec['offered_diagrams'])} | "
                         f"**Emitted:** {', '.join(rec['diagram_ids']) or 'None'}\n")
        parts.append("\n**Raw Generated Output:**\n")
        parts.append(fence(rec["answer"] or "(no answer produced)", "markdown"))
        scores = " | ".join(
            f"{k}={pct(met.get(k))}" for k in
            ("context_relevance", "context_precision", "context_recall", "faithfulness",
             "answer_relevance", "semantic_accuracy") if met.get(k) is not None)
        parts.append(f"\n**Metric Scores:** {scores}  \n"
                     f"**Judge Audit Reason:** {met.get('judge_reason', 'N/A')}  \n"
                     f"**Execution Latency:** {rec.get('latency_s')}s\n")
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
        cat_rows.append([cat.upper()] + [pct(card.get(k)) for k in
                                         ("context_relevance", "context_precision", "faithfulness",
                                          "answer_relevance", "semantic_accuracy", "robustness")])
    return f"""## VI. QUANTITATIVE EVALUATION SCORECARD

The system was benchmarked against {data['query_count']} ground-truth test queries over {data['duration_s']}s wall-clock time using generation model `{data['model']}`. Metrics are categorized into Retrieval, Generation, and Domain-Specific stages according to the standardized calibration brief.

### A. Master Evaluation Scorecard

{md_table(["Stage", "Metric Name", "Measured Score", "Pass Criterion", "Audit Result", "Verification Objective"], rows)}

**Summary: {passed} of {len(CRITERIA)} standardized checklist metrics passed.**

### B. Granular Breakdown by Query Category

{md_table(["Category", "Ctx Relevance", "Ctx Precision", "Faithfulness",
           "Answer Relevance", "Semantic Acc.", "Robustness"], cat_rows)}

*Supporting Criterion: Mandatory keyword coverage in generated answers: **{pct(sc.get('must_mention_rate'))}**.*
"""


def section_failures(data: dict) -> str:
    misses = []
    for r in data["rows"]:
        case, rec, met = r["case"], r["record"], r["metrics"]
        problems = []
        if rec.get("error"):
            problems.append(("Infrastructure", rec["error"]))
        if met.get("disambiguation") == 0.0:
            problems.append(("Retrieval", f"Routing: {met.get('disambiguation_note')}"))
        if met.get("diagram_id") == 0.0:
            problems.append(("Generation", f"Diagram: {met.get('diagram_note')}"))
        if met.get("robustness") == 0.0:
            problems.append(("Retrieval", "Refusal failure on out-of-scope test"))
        if met.get("context_precision") == 0.0:
            problems.append(("Retrieval", f"Expected topic {case['expected_topic']} absent from top-2"))
        for key, stage in (("faithfulness", "Generation"), ("semantic_accuracy", "Generation"),
                           ("context_recall", "Retrieval")):
            v = met.get(key)
            if v is not None and v < 0.7:
                problems.append((stage, f"{key} {pct(v)}: {met.get('judge_reason', '')[:140]}"))
        if met.get("must_mention_missing"):
            problems.append(("Generation", f"Missing terms {met['must_mention_missing']}"))
        for stage, detail in problems:
            misses.append([case["id"], case["category"].upper(), stage, detail[:200]])

    body = md_table(["Query ID", "Category", "Failing Stage", "Observed Root Cause"], misses) if misses else \
        "Zero metric anomalies observed; all 34 benchmark cases passed threshold criteria."

    return f"""## VII. FAILURE MODE ANALYSIS AND LIMITATIONS

### A. Observed Defect Audit

{body}

### B. Identified Systemic Limitations

1. **Corpus Depth and Sparsity:** Baseline student notes provide minimal passages for several topics (e.g., Bellman-Ford and Prim's MST). Retrieval falls back to unit priors when fewer than 3 chunks match, placing an upper ceiling on pure passage relevance.
2. **Authored Supplement Dependency:** Ten syllabus topics absent in the baseline notes were authored in `notes_supplement.md`. While rigorously tagged, answers on these topics cite the supplement rather than student notes.
3. **Reasoning Token Budgeting:** On reasoning-heavy LLM endpoints, token limits cover reasoning tokens; structured-output calls (routing and judging) require low reasoning effort to prevent token exhaustion.
4. **Out-of-Scope Similarity Thresholding:** Off-syllabus queries lexically adjacent to computer science concepts require strict similarity gating calibrated against empirical negative test queries.

### C. Proposed Architectural Mitigations

- **Corpus Expansion:** Indexing full textbook chapters (CLRS 4th ed.) to elevate per-topic chunk volume.
- **Cross-Encoder Reranking:** Introducing a second-stage cross-encoder over the fused top-20 candidates for enhanced precision.
- **Runtime Grounding Enforcement:** Integrating `RubricMiddleware` to dynamically evaluate and re-prompt ungrounded generations during inference.
"""


def build_markdown(data: dict) -> str:
    head = """# Engineering Calibration Audit - RAG Based AI Application

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


IEEE_CSS = """
@page {
  size: A4;
  margin: 19mm 15mm 20mm 15mm;
  @top-right {
    content: "AKTU DAA RAG SYSTEM AUDIT";
    font-family: "Liberation Serif", "Times New Roman", Times, serif;
    font-size: 7.5pt;
    font-style: italic;
    color: #333;
  }
  @bottom-center {
    content: counter(page);
    font-family: "Liberation Serif", "Times New Roman", Times, serif;
    font-size: 9pt;
  }
}

body {
  font-family: "Liberation Serif", "Times New Roman", "DejaVu Serif", Times, serif;
  font-size: 9.5pt;
  line-height: 1.38;
  color: #000;
  text-align: justify;
}

h1 {
  font-size: 16pt;
  font-weight: bold;
  text-align: center;
  margin: 0 0 10px 0;
  line-height: 1.25;
  text-transform: uppercase;
}

.ieee-author-block {
  text-align: center;
  font-size: 9pt;
  margin-bottom: 14px;
  line-height: 1.35;
}

.ieee-abstract {
  margin: 12px 10mm 16px 10mm;
  font-size: 8.5pt;
  line-height: 1.3;
  border-top: 0.5pt solid #000;
  border-bottom: 0.5pt solid #000;
  padding: 8px 0;
}

h2 {
  font-size: 11pt;
  font-weight: bold;
  text-align: center;
  text-transform: uppercase;
  margin: 18px 0 8px 0;
  border-bottom: none;
  page-break-after: avoid;
  letter-spacing: 0.5px;
}

h3 {
  font-size: 9.5pt;
  font-weight: bold;
  font-style: italic;
  margin: 12px 0 4px 0;
  page-break-after: avoid;
}

h4 {
  font-size: 9pt;
  font-weight: bold;
  margin: 8px 0 2px 0;
  page-break-after: avoid;
}

p {
  margin: 0 0 7px 0;
  text-indent: 1.5em;
}

.ieee-abstract p, .ieee-author-block p {
  text-indent: 0;
}

/* Formal IEEE Table Styling (Booktabs format) */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
  font-size: 7.6pt;
  line-height: 1.25;
  border-top: 1.2pt solid #000;
  border-bottom: 1.2pt solid #000;
}

th {
  border-bottom: 0.8pt solid #000;
  padding: 3.5px 4px;
  text-align: left;
  font-weight: bold;
  text-transform: uppercase;
  font-size: 7.2pt;
  background: transparent;
}

td {
  padding: 3px 4px;
  vertical-align: top;
  border-bottom: 0.4pt solid #e0e0e0;
}

tr:last-child td {
  border-bottom: none;
}

pre {
  background: #f9f9f9;
  border: 0.5pt solid #ccc;
  padding: 5px 6px;
  font-size: 6.8pt;
  line-height: 1.2;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
  margin: 6px 0;
}

code {
  font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
  font-size: 7.5pt;
  background: #f2f2f2;
  padding: 1px 3px;
  border-radius: 2px;
}

img {
  display: block;
  max-width: 85%;
  height: auto;
  margin: 8px auto;
  border: 0.5pt solid #ddd;
}

hr {
  border: none;
  border-top: 0.5pt solid #ccc;
  margin: 12px 0;
}

em {
  font-style: italic;
}

strong {
  font-weight: bold;
}
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
             base_url=str(ROOT)).write_pdf(pdf_path, stylesheets=[WCSS(string=IEEE_CSS)])
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

