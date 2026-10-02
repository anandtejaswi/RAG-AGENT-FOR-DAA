# AKTU DAA RAG System

A retrieval augmented generation system for the AKTU subject **Design and
Analysis of Algorithms (KCS-503)**, built for task `CALIB-RAG-AKTU-DAA-01`.

It answers syllabus questions with step-by-step derivations, computes numerical
answers with exact solvers rather than with the language model, references
diagram assets by persistent ID, resolves cross-topic keyword collisions, and
refuses out-of-scope questions. The deliverable report is produced by the same
pipeline that answers queries.

---

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install sentence-transformers rank_bm25 pymupdf deepagents \
    langchain-google-genai langchain-openai langchain-text-splitters \
    python-dotenv markdown weasyprint matplotlib

cp model_details.txt .env          # or write the variables below yourself

.venv/bin/python cli.py ingest     # build chunk index + embeddings (~1 min)
.venv/bin/python cli.py diagrams   # render the 21 diagram assets
.venv/bin/python cli.py ask "Solve T(n) = 2T(n/2) + n by the Master Theorem"
.venv/bin/python cli.py eval       # run the 34 labelled test queries (~7 min)
.venv/bin/python cli.py report     # write out/report.md and out/report.pdf
.venv/bin/python cli.py serve      # run the FastAPI Agent Server (http://localhost:8000)
.venv/bin/python cli.py ui         # start Agent Server + assistant-ui Frontend (http://localhost:5173)
```

`cli.py ingest` and `cli.py diagrams` need no API key. `ask`, `eval`, `serve`, `ui` and the
report do.

## Configuration

`.env` at the project root:

```
MUNSHI_MODEL_PROVIDER=openai_compat
MUNSHI_MODEL=z-ai/glm-5.3-flash
MUNSHI_MODEL_BASE_URL=https://openrouter.ai/api/v1
MUNSHI_MODEL_API_KEY=<your key>
```

To use Gemini instead, set `MUNSHI_MODEL_PROVIDER=google_genai`,
`MUNSHI_MODEL=gemini-2.5-flash` and put the key in `MUNSHI_MODEL_API_KEY`. No
code changes are needed. Embeddings are always the local open-source
`BAAI/bge-base-en-v1.5`, which runs on CPU and needs no key.

## How it works

```
sources -> heading-aware chunks -> unit/topic tags -> dense + BM25 index
query -> topic routing -> hybrid retrieval + topic priority -> top passages
      -> diagram lookup -> out-of-scope gate -> Deep Agent (+ 9 solvers) -> answer
```

Retrieval completes before generation begins, so the passages and scores in the
report are exactly what the model received. The agent's only tools are the
numeric solvers: it explains, the solvers compute.

| Path | Role |
|---|---|
| `rag/ingest.py` | chunking, syllabus tagging, embeddings |
| `rag/retrieve.py` | query routing, hybrid search, diagram lookup, refusal gate |
| `rag/solvers.py` | nine exact solvers (Master Theorem, knapsack, LCS, matrix chain, Floyd-Warshall, Dijkstra, Bellman-Ford, recursion tree, fractional knapsack) |
| `rag/answer.py` | Deep Agent, prompt, diagram ID validation |
| `rag/evaluate.py` | the nine checklist metrics over `data/testset.json` |
| `rag/report.py` | seven-section evaluation report as Markdown and PDF |
| `tools/make_diagrams.py` | renders the diagram assets and their ID registry |

## Data

| Path | Contents |
|---|---|
| `data/1038678511-Daa-Aktu-Notes.txt` | supplied study notes |
| `data/syllabus-daa` | official KCS-503 syllabus PDF |
| `data/pyq/` | eight previous year question papers |
| `data/notes_supplement.md` | authored notes covering syllabus topics the supplied notes omit |
| `data/syllabus.json` | unit and topic map, 29 topics, drives tagging and routing |
| `data/diagrams/` | 21 PNG assets plus `diagrams.json`, the persistent ID registry |
| `data/testset.json` | 34 labelled queries; 11 taken verbatim from the question papers |

Diagram IDs follow `DIAG_DAA_U{unit}_{CONCEPT}_{NN}`, for example
`DIAG_DAA_U2_AVL_ROT_01`. An ID the retriever did not offer is stripped from the
answer and logged, so a broken reference cannot reach a user interface.

## Tests

```bash
.venv/bin/python tests/test_solvers.py    # known-answer checks for all nine solvers
.venv/bin/python tests/test_retrieve.py   # collision pairs route to the right topic
```

`tests/test_retrieve.py` needs the index built; it skips the model-dependent
checks when no API key is present.

## Output

`cli.py eval` writes `out/results.json` with the full per-query log and the
scorecard. `cli.py report` renders `out/report.md` and `out/report.pdf` with the
seven sections the task brief requires.

`run_log.md` records the engineering decisions, what testing found, and how each
defect was fixed.
