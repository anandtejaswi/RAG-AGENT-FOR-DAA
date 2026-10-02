# AKTU DAA RAG System

An examiner-grade Retrieval-Augmented Generation (RAG) system for the AKTU subject **Design and Analysis of Algorithms (KCS-503)**, built for task `CALIB-RAG-AKTU-DAA-01`.

The system produces step-by-step mathematical proofs and algorithmic derivations, computes all numerical answers using **9 exact deterministic Python solvers** rather than probabilistic LLM arithmetic, links 21 persistent visual diagram schematics, disambiguates cross-unit syllabus keyword collisions, and strictly enforces out-of-scope refusals.

It includes a self-deployed **FastAPI Agent Server** with real-time SSE streaming, an **assistant-ui React frontend**, native **LangSmith / LangChain Tracing V2** telemetry, and an automated **IEEE-formatted PDF audit report generator**.

---

## Quick Start & Running Commands

### 1. Environment Setup

```bash
# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install Python dependencies
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install sentence-transformers rank_bm25 pymupdf deepagents \
    langchain-google-genai langchain-openai langchain-text-splitters \
    python-dotenv markdown weasyprint matplotlib fastapi uvicorn langsmith

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Configure Environment Variables

Create `.env` in the root directory (or copy `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```env
# Model Configuration (e.g. OpenRouter, OpenAI, vLLM)
LLM_PROVIDER=openai_compat
LLM_MODEL=z-ai/glm-5.3-flash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_model_api_key_here

# LangSmith / LangChain Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=rag-aktu
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

---

## System CLI Commands

All primary functions can be invoked directly from the command line using `cli.py`:

| Command | Description |
|---|---|
| `.venv/bin/python cli.py ingest` | Parses notes, syllabus, and PYQ PDFs, splits text into tagged chunks, and builds the dense (`BAAI/bge-base-en-v1.5`) & BM25 indexes. |
| `.venv/bin/python cli.py diagrams` | Generates all 21 visual diagram assets in `data/diagrams/` and compiles the JSON catalog. |
| `.venv/bin/python cli.py ask "your query"` | Answers an individual student query end-to-end with retrieval provenance, tool calls, and latency. |
| `.venv/bin/python cli.py eval [category]` | Runs the full 34-query metric-gated benchmark suite across analytical, DP, graph, and collision test cases. |
| `.venv/bin/python cli.py report` | Re-compiles `out/report.md` and generates the formal IEEE-style `out/report.pdf` evaluation report. |
| `.venv/bin/python cli.py serve [port]` | Starts the FastAPI Agent Server with SSE streaming and diagram mounting (default: `http://localhost:8000`). |
| `.venv/bin/python cli.py ui` | Launches both the FastAPI Agent Server and the `assistant-ui` frontend simultaneously. |

---

## Running the Web UI & Agent Server

To run the interactive web application:

```bash
# Terminal 1: Start FastAPI Agent Server
.venv/bin/python server.py
# Server runs at http://localhost:8000 (API docs at /docs)

# Terminal 2: Start assistant-ui React Frontend
cd frontend
npm run dev
# Frontend accessible at http://localhost:5173
```

---

## How It Works: Architectural Overview

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Dual-Engine Topic Classification & Intent Routing                  │
│    ├─ LLM Structured Router (GLM-5.3-flash, reasoning_effort="low")    │
│    ├─ Deterministic Lexical Regex Classifier (word-boundary & priors)  │
│    └─ Outputs: Unit, Topic ID, Confidence (0..1), Solver, Diagram Req  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Dual Hybrid Retrieval & Reciprocal Rank Fusion (RRF)                │
│    ├─ Dense Semantic: BAAI/bge-base-en-v1.5 (768-dim normalized)       │
│    ├─ Sparse Lexical: BM25Okapi over section headings + body text      │
│    └─ RRF Fusion (k=60) + Topic Priority Ranking (1.5x on-topic boost)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Out-of-Scope Similarity Floor Gate                                  │
│    • Cosine floor: 0.45 for topic-matched / 0.58 for unclassified      │
│    • Low-similarity queries trigger deterministic refusal messages     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. LangChain Deep Agent (+ 9 Deterministic Python Solvers)             │
│    • Solvers: Master Theorem, recursion tree, 0/1 knapsack, fractional │
│      knapsack, LCS, matrix chain, Floyd-Warshall, Dijkstra, Bellman-   │
│      Ford (reproducing exact matrices & tables).                       │
│    • Grounding: Mandatory inline citations [C0042], diagram links.     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Post-Verification & Real-Time SSE Streaming                         │
│    • Strips unoffered/hallucinated diagram IDs                         │
│    • Streams formatted markdown & telemetry to assistant-ui            │
│    • Exports telemetry graphs to LangSmith (LANGCHAIN_TRACING_V2)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── cli.py                     # Unified CLI entry point
├── server.py                  # FastAPI Agent Server with SSE streaming
├── frontend/                  # assistant-ui React + TypeScript + Tailwind UI
├── rag/
│   ├── config.py              # Model providers, environment & LangSmith tracing setup
│   ├── ingest.py              # Multi-tier chunking, section extraction & embeddings
│   ├── retrieve.py            # Dual classifier, dense+BM25 hybrid search, RRF & gating
│   ├── solvers.py             # 9 exact deterministic algorithm solvers
│   ├── answer.py              # LangChain Deep Agent, prompt directives, verification
│   ├── evaluate.py            # 9-metric evaluator over 34 benchmark test cases
│   └── report.py              # IEEE Markdown & PDF report generator
├── data/
│   ├── 1038678511-Daa-Aktu-Notes.txt # Supplied study notes
│   ├── notes_supplement.md    # Authored gap-filling notes for 100% syllabus coverage
│   ├── syllabus-daa           # Official KCS-503 curriculum PDF
│   ├── syllabus.json          # 5-unit, 29-topic hierarchy and keyword taxonomy
│   ├── diagrams/              # 21 generated PNG schematics + diagrams.json registry
│   └── testset.json           # 34 labelled benchmark queries
├── tests/
│   ├── test_solvers.py        # Known-answer numerical verification test suite
│   └── test_retrieve.py       # Topic disambiguation & collision verification
├── out/
│   ├── results.json           # Complete benchmark evaluation logs & scorecard
│   ├── report.md              # Generated Markdown audit report
│   └── report.pdf             # Compiled IEEE PDF audit report
├── .env.example               # Template environment configuration file
└── README.md                  # System documentation
```

---

## Benchmark Metrics & Evaluation

Run the evaluation suite:
```bash
.venv/bin/python cli.py eval
```

The system is evaluated against the 9 standardized checklist criteria:
1. **Context Relevance ($\ge 60\%$)**
2. **Context Recall ($\ge 80\%$)**
3. **Context Precision ($\ge 90\%$)**
4. **Faithfulness / Groundedness ($\ge 80\%$)**
5. **Answer Relevance ($\ge 85\%$)**
6. **Semantic & Mathematical Accuracy ($\ge 90\%$)**
7. **Cross-Topic Disambiguation Rate ($\ge 90\%$)**
8. **Diagram ID Linkage Rate ($\ge 90\%$)**
9. **Negative / Out-of-Scope Robustness ($\ge 90\%$)**
