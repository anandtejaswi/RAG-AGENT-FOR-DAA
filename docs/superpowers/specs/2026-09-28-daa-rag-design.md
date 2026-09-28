# AKTU DAA RAG — Design Spec

Date: 2026-09-28
Task: CALIB-RAG-AKTU-DAA-01 (see `AKTU_DAA_RAG_Calibration_Task_Brief.pdf`)

## 1. Goal

A working Retrieval-Augmented Generation system over the AKTU *Design and Analysis of Algorithms* syllabus, plus a formal PDF evaluation report. The report is the graded deliverable; the system exists to produce its evidence.

Success means the report can honestly check every box of the brief's 9-metric checklist with logged evidence: ≥10 query/response logs, ≥3 numerical walk-throughs, a disambiguation stress test, a diagram ID index, a scorecard, and a failure-mode analysis.

## 2. Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Generation LLM | Google Gemini via `langchain-google-genai` | User choice |
| Orchestration | LangChain `deepagents` (`create_deep_agent`) with solvers as tools | User choice; retrieval stays outside the agent so scores are logged exactly |
| Embeddings | `BAAI/bge-base-en-v1.5` via `sentence-transformers`, CPU | Open-source, strong on short technical text, no service needed |
| Corpus | User-provided PDFs/notes in `data/pdfs/` | User choice |
| Vector store | numpy matrix + JSONL, no DB | Corpus is a few hundred chunks; a DB adds nothing |
| Sparse retrieval | `rank_bm25` | Cheap, catches exact terms (e.g. "Bellman-Ford") dense misses |
| Numerics | Python solvers exposed as Gemini function-calling tools | LLM arithmetic is not trustworthy; the brief demands exact tables |
| Disambiguation | Query classification → topic metadata filter | Solves keyword collision at retrieval, where the brief says it must be solved |
| Report | Markdown template → PDF (pandoc if present, else weasyprint) | Zero design effort, fully reproducible |

## 3. Layout

```
rag-aktu/
  data/pdfs/              user PDFs (input, not committed if large)
  data/diagrams/          image files + diagrams.json registry
  data/syllabus.json      unit → topic → keywords (AKTU KCS-503, 5 units)
  data/testset.json       ~25 labelled queries
  index/                  chunks.jsonl + embeddings.npy + bm25.pkl (built)
  rag/ingest.py           pdf → chunks → index
  rag/retrieve.py         classify + hybrid search + topic filter + diagram lookup
  rag/solvers.py          master_theorem, knapsack, lcs, floyd_warshall, mcm, dijkstra
  rag/answer.py           Gemini call with tools; ask(query) → AnswerRecord
  rag/evaluate.py         run testset → results.json (logs + scorecard)
  rag/report.py           results.json → report.md → report.pdf
  cli.py                  ingest | ask "…" | eval | report
  tests/test_solvers.py   known-answer checks per solver
  tests/test_retrieve.py  disambiguation pairs hit expected topic
  README.md               run instructions (required by brief)
  .env.example            GOOGLE_API_KEY=
```

## 4. Data model

### 4.1 Syllabus map (`data/syllabus.json`)
```json
{ "units": [
  { "unit": 1, "name": "Introduction & Asymptotics",
    "topics": [
      { "topic": "asymptotic_notation", "keywords": ["big-o","omega","theta","growth"] },
      { "topic": "recurrences", "keywords": ["master theorem","recursion tree","substitution"] },
      { "topic": "sorting", "keywords": ["heap sort","quick sort","merge sort","shell sort"] }
    ] },
  ...
]}
```
Five units following the AKTU DAA syllabus: (1) asymptotics, recurrences, sorting; (2) advanced data structures: RB trees, B-trees, binomial/Fibonacci heaps, tries, skip lists; (3) divide and conquer, greedy (activity selection, fractional knapsack, Huffman, MST, shortest paths); (4) dynamic programming (0/1 knapsack, LCS, MCM, Floyd-Warshall), backtracking, branch and bound; (5) string matching, NP-completeness, approximation, randomized algorithms. Keywords are hand-written; the file is the single source of truth for the classifier enum and the index-time tagging.

### 4.2 Chunk (`index/chunks.jsonl`)
```json
{ "id": "C0042", "unit": 3, "topic": "shortest_paths", "source": "notes.pdf", "page": 57,
  "heading": "Dijkstra's Algorithm", "text": "..." }
```
Chunking: split on detected headings (font-size jump or numbered heading regex via pymupdf spans); any segment over ~350 tokens is windowed at 350 with 50 overlap. Unit/topic assigned by keyword-hit count against `syllabus.json`; tie broken by the PDF's chapter position if the PDF has a table of contents, else by the previous chunk's topic (topics are contiguous in notes).

### 4.3 Diagram registry (`data/diagrams/diagrams.json`)
```json
{ "id": "DIAG_DAA_U2_AVL_ROT_01", "unit": 2, "topic": "balanced_trees",
  "concept": "AVL left-left rotation", "keywords": ["avl","rotation","ll","balance factor"],
  "file": "avl_rot_01.png", "caption": "Single right rotation restoring balance" }
```
ID format: `DIAG_DAA_U{unit}_{CONCEPT}_{NN}`. Curated by hand from what the PDFs contain: images are page-region crops (pymupdf `get_pixmap(clip=…)`) or Graphviz-generated when the notes have no figure for a required concept (brief lists: recurrence trees, AVL rotations, RB rebalancing, graph traversals, MST stages, DP tables, flowcharts). Target 15–25 entries.

### 4.4 Test set (`data/testset.json`)
```json
{ "id": "Q07", "category": "disambiguation", "query": "Explain edge relaxation in Bellman-Ford",
  "expected_unit": 3, "expected_topic": "shortest_paths",
  "expected_diagram_ids": [], "expected_numeric": null,
  "must_mention": ["negative", "V-1"] }
```
Categories and minimum counts: `numeric` ×4, `disambiguation` ×6 (3 collision pairs), `diagram` ×4, `oos` ×3, `general` ×6. `expected_numeric` for numeric queries holds the solver's exact expected output (e.g. `{"case": 2, "bound": "Theta(n log n)"}` or a knapsack table).

### 4.5 AnswerRecord (return of `ask()`)
```python
{ "query", "classification": {unit, topic, confidence, needs_diagram, numeric_task},
  "chunks": [{id, score, unit, topic, text}],   # top-5 after fusion
  "diagram_ids": [...], "tool_calls": [{name, args, result}],
  "answer": str, "refused": bool, "latency_s": float }
```

## 5. Retrieval flow (`retrieve.py`)

1. **Classify** — one Gemini call with a JSON schema: `unit` (1–5 or null), `topic` (enum from syllabus.json or null), `confidence` (0–1), `needs_diagram` (bool), `numeric_task` (enum of solver names or null). Null topic with low confidence means out of scope.
2. **Hybrid search** — dense cosine over `embeddings.npy` and BM25 each return top-20. Reciprocal rank fusion, k=60.
3. **Topic filter** — if `confidence ≥ 0.7`, restrict candidates to the classified topic (fall back to unit if fewer than 3 chunks match). Otherwise multiply scores of matching-topic chunks by 1.5. Return top-5.
4. **Diagram lookup** — BM25 over `concept + keywords` of the registry, restricted to classified topic (or unit). Return up to 2 IDs when `needs_diagram` is true.
5. **OOS gate** — refuse without generation if topic is null, or the best fused score is below a threshold tuned once on the testset OOS queries (constant in `retrieve.py`, `ponytail:` comment).

## 6. Generation (`answer.py`, `solvers.py`)

Orchestrated with LangChain `deepagents`. Retrieval is **not** an agent tool: it runs before the agent (Section 5) so the retrieved chunks and scores are deterministic and logged for the retrieval-stage metrics. The agent only reasons over the supplied context and calls the numeric solvers.

- Model: `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)` (constant in `answer.py`; `gemini-2.5-pro` if flash proves weak on proofs). The classifier in Section 5 uses the same model with `.with_structured_output(Classification)`.
- Agent: `create_deep_agent(model=llm, tools=SOLVER_TOOLS, system_prompt=SYSTEM_PROMPT)`. No sub-agents, no filesystem tools beyond the defaults; the built-in planning tool may be used by the model but is not required.
- System prompt: answer only from the supplied context; cite chunk IDs inline as `[C0042]`; when a diagram is supplied, emit `[Diagram: DIAG_…]` exactly once; for any numeric task call the matching solver tool and reproduce its output step by step; if the context is insufficient say so instead of guessing.
- User message: the query, followed by a context block with the top-5 chunks (id + text) and the matched diagram entries (id + caption).
- Tools: `solvers.py` functions wrapped with `langchain_core.tools.tool`. Each returns a structured dict the model must reproduce:
  - `master_theorem(a, b, f_exponent, log_power)` → case, comparison, bound
  - `knapsack_01(weights, values, capacity)` → full DP table, chosen items
  - `lcs(x, y)` → length table, direction arrows, one LCS
  - `floyd_warshall(matrix)` → D^(k) for each k
  - `matrix_chain(dims)` → m and s tables, parenthesization
  - `dijkstra(graph, source)` → per-iteration distance/visited trace
  - `recursion_tree(a, b, f_exponent, depth)` → per-level cost, total
- Invocation: `agent.invoke({"messages": [HumanMessage(...)]}, config={"recursion_limit": 12})`. Final text is the last `AIMessage`. Tool calls are recovered from the returned message list: each `AIMessage.tool_calls` entry paired with the following `ToolMessage` by `tool_call_id`, giving `{name, args, result}` for the log.
- Post-check: any `DIAG_` token in the answer must be in the supplied diagram list; strip and log any that is not (counts as a linkage failure in eval).

## 7. Evaluation (`evaluate.py`)

Runs every testset query through `ask()` and computes the 9 metrics:

| Metric | Method |
|---|---|
| Context Relevance | fraction of top-5 chunks whose topic == expected_topic (label-based) |
| Context Recall | Gemini judge: does the context contain every fact needed? 0–1 |
| Context Precision | 1 if top-1 or top-2 chunk has expected_topic, else 0 |
| Faithfulness | Gemini judge: every claim supported by context? 0–1 |
| Answer Relevance | Gemini judge: directly answers all parts? 0–1; plus `must_mention` hit rate |
| Semantic/Math Accuracy | numeric: tool result == expected_numeric and answer reproduces it; else judge |
| Disambiguation Rate | classified topic == expected_topic on disambiguation queries |
| Diagram ID Rate | emitted IDs == expected_diagram_ids on diagram queries |
| OOS Robustness | refused == true on oos queries, and no refusal on in-scope queries |

Judge prompts are fixed rubrics returning JSON `{score, reason}`. Output `results.json` = per-query AnswerRecord + metric values + scorecard averages by category.

## 8. Report (`report.py`)

Markdown template with the brief's seven sections in order, filled from `results.json` and `diagrams.json`:
1. Architecture overview (static text + actual chunk/diagram counts)
2. Diagram & Asset ID Index (table from registry)
3. Numerical Handling Demonstration (all `numeric` queries: prompt, tool call, generated steps)
4. Cross-Topic Disambiguation Stress Test (collision pairs side by side: classification, top-3 chunks, answer excerpt)
5. End-to-End Test Case Logs (every query: prompt, chunks with scores, raw output)
6. Quantitative Scorecard (9 rows, pass/fail against brief's criteria)
7. Failure Mode Analysis (auto-listed misses, tagged *retrieval* if expected topic absent from top-5 else *generation*; hand-written mitigation notes appended)

Conversion: `pandoc report.md -o report.pdf` if pandoc exists, else `markdown` → `weasyprint`.

## 9. Testing

- `tests/test_solvers.py`: one known-answer assert per solver (e.g. T(n)=2T(n/2)+n → Θ(n log n); knapsack classic example → 220).
- `tests/test_retrieve.py`: the three collision pairs classify and retrieve to their expected topics (requires built index and API key; skipped otherwise).
- Everything else is validated by `cli.py eval`.

## 10. Dependencies

`pymupdf`, `sentence-transformers`, `rank_bm25`, `numpy`, `deepagents`, `langchain-google-genai`, `python-dotenv`, `weasyprint` (or system pandoc). LangChain retrievers/vector stores are not used; the retriever is plain numpy + rank_bm25 as in Section 5.

## 11. Out of scope

No web UI, no vector DB, no reranker, no multi-hop retrieval, no OCR of scanned PDFs (if a PDF has no text layer, it is reported as unsupported). No streaming.

## 12. Open items requiring user input

- PDFs must be placed in `data/pdfs/` before ingest.
- `GOOGLE_API_KEY` in `.env` (name expected by `langchain-google-genai`).
- Diagram registry curation happens after the PDFs are seen; concept list will be confirmed then.

## 13. Alignment with LangChain documentation (verified 2026-09-28 via docs.langchain.com MCP)

| Spec strategy | Docs verdict | Source page | Action |
|---|---|---|---|
| Retrieval before generation, agent only reasons over supplied context | **Documented architecture: "2-Step RAG"** (high control, predictable latency, use case "FAQs, documentation bots"). With the judge-based validation in §7 it matches the "Hybrid" row ("domain-specific Q&A with quality validation"). | `oss/python/deepagents/retrieval` → RAG architectures table | Keep. |
| `create_deep_agent(model=…, tools=…, system_prompt=…)` | Matches signature. Docs pass model as a `"provider:model"` string (`google_genai:gemini-…`), not a `ChatGoogleGenerativeAI` instance. | `deepagents/models`, `deepagents/tools`, `deepagents/rag` | **Change:** use `model="google_genai:<id>"`. |
| `gemini-2.5-flash` | Not in the suggested list. Docs suggest `gemini-3.6-flash` (82% on deep-agents eval suite) and `gemini-3.1-pro-preview`. | `deepagents/models` → Suggested models | **Change:** default `google_genai:gemini-3.6-flash`. |
| Solvers as `@tool` functions | Documented: plain functions or `@tool` from `langchain.tools`, schema inferred from signature + docstring. | `deepagents/tools` → Custom tools | Keep. Use `@tool(parse_docstring=True)` as the RAG tutorial does. |
| "Built-in planning tool may be used" | **Wrong as written.** Since deepagents 0.7 task planning is opt-in via `TodoListMiddleware`; filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) are on by default. | `deepagents/overview` → Task planning, Execution environment | **Change:** no todo middleware; hide filesystem tools via a harness profile with `excluded_tools` so the agent sees only the solvers. |
| Classifier via `.with_structured_output(Classification)` | Documented on chat models; `method="json_schema"` is the provider-native path and is what the LangSmith RAG tutorial uses for graders. | `langchain/models` → Structured output; `langsmith/evaluate-rag-tutorial` | Keep; pass `method="json_schema"`. |
| Tool-call logging from `AIMessage.tool_calls` + `ToolMessage.tool_call_id` | Documented message fields. | `langchain/messages` | Keep. |
| `recursion_limit` in `config` | Documented (`config.recursion_limit`); default is adequate for tool-only agents. | `deepagents/frontend/subagent-streaming` | Keep as a safety cap. |
| Heading-based chunking, ~350 tokens / 50 overlap | Docs recommend `RecursiveCharacterTextSplitter` (default example 1000 chars / 200 overlap) as "the recommended text splitter for generic text". No guidance against heading splits. | `langchain/knowledge-base`, `deepagents/rag` | **Change:** use `RecursiveCharacterTextSplitter` from `langchain-text-splitters` for the windowing step; keep heading boundaries and metadata tagging (our own logic). |
| PDF loading via pymupdf | Tutorial uses `pypdf`; either is a documented loader. | `langchain/knowledge-base` | Keep pymupdf (needed for page-region crops anyway). |
| Hybrid dense + BM25 with RRF, topic metadata filter | **Not a documented deepagents pattern.** Docs only show vector `similarity_search(k=4)`; hybrid search appears only as vendor features (MongoDB, Elasticsearch). | `integrations/retrievers` | Keep, but the report must present it as our own retrieval design, not a LangChain-recommended recipe. |
| Gemini judge for recall / faithfulness / relevance | Matches LangSmith RAG eval tutorial: four LLM-as-judge evaluators (correctness vs reference, relevance vs input, groundedness vs retrieved docs, retrieval relevance) with structured-output graders. | `langsmith/evaluate-rag-tutorial` | Keep; reuse the tutorial's grader prompt structure. Runs offline, no LangSmith account required. |
| Runtime grounding check | Docs offer `RubricMiddleware` (beta, deepagents ≥0.6.5): grader sub-agent re-prompts until a rubric passes. | `deepagents/rubric` | Optional; not adopted (adds LLM calls per query; §7 judge already measures faithfulness). Mention in report as a mitigation path. |
| `GOOGLE_API_KEY` | Confirmed as the Gemini env var in the RAG tutorial setup. | `deepagents/rag` → Setup | Keep. |

Sections 2, 6 and 10 are superseded where this table says **Change**.
