# Run Log

Engineering log for CALIB-RAG-AKTU-DAA-01. Records what was built, the decisions
taken and why, what testing found, and what was changed in response.

---

## 1. Inputs and environment

**Supplied material** found in `data/`:

| Item | Form | Note |
|---|---|---|
| `1038678511-Daa-Aktu-Notes.txt` | 26 KB plain text | Study notes, numbered sections, no figures |
| `syllabus-daa` | 1 page PDF | Official AKTU KCS-503 detailed syllabus |
| `pyq/*.pdf` | 8 PDFs, 2 pages each | Previous year question papers, 2017-18 to 2024-25 |

**Environment.** Python 3.14 on Fedora. Verified before committing to the stack
that `torch` and `sentence-transformers` publish cp314 wheels; they do
(torch 2.14.0+cpu). All work happens in a project-local `.venv`.

**Model.** Supplied via `model_details.txt`: `z-ai/glm-5.3-flash` through
OpenRouter's OpenAI-compatible endpoint. That file contains a live API key, so it
was added to `.gitignore` and copied to `.env`, which is also ignored. The model
layer in `rag/config.py` reads the provider from the environment, so switching to
Gemini is a configuration change rather than a code change.

---

## 2. Decisions taken and the reasoning

**Retrieval runs before generation, not as an agent tool.** The brief requires the
retrieval stage to be scored independently of generation. Putting retrieval inside
the agent would make the retrieved set depend on the model's whim and would make
the logged scores irreproducible. The LangChain documentation names this the
"2-Step RAG" architecture and recommends it for exactly this control property.

**Hybrid retrieval with a syllabus topic layer.** Dense embeddings alone confuse
topics that share vocabulary. The pipeline runs a dense bi-encoder
(`BAAI/bge-base-en-v1.5`) and BM25 in parallel, fuses them with reciprocal rank
fusion, then applies the syllabus topic predicted for the query. The topic layer
is what actually resolves the collisions the brief asks about; fusion alone does
not.

**Two routers, not one.** A deterministic keyword router scores the query against
the syllabus map, and the language model routes it as well. Agreement raises
confidence; when the model returns no topic and the keyword evidence is weak, the
query is treated as out of scope. This keeps the system usable when the API is
unavailable and gives the out-of-scope gate a second, independent signal.

**Arithmetic is delegated to code, never to the model.** Nine solvers compute
recurrences, DP tables, and graph traces. They are exposed to the Deep Agent as
tools. The model narrates; the solver computes. This is the only way to honestly
claim mathematical accuracy.

**Diagrams are generated from the same solvers.** The supplied notes contain no
figures, so 21 assets were rendered programmatically. Every figure that shows a
computed table or trace is produced by calling the same solver the agent calls, so
a figure and an answer cannot disagree.

**Question papers are indexed but excluded from answer context.** They contain
questions, not explanations. Including them would damage faithfulness. They are
used for provenance and as the source of authentic test queries: 11 of the 34 test
questions are taken verbatim from the supplied papers.

**Gaps in the supplied notes were filled explicitly.** The notes omit ten topics
that the official syllabus prescribes: LCS detail, Floyd-Warshall, backtracking,
branch and bound, tries, skip lists, linear-time sorting, randomized algorithms,
FFT, and job sequencing. They were authored as `data/notes_supplement.md` and
tagged with their own source so any answer citing them is traceable. Without this
the system could not answer a large part of its own syllabus.

---

## 3. Build order

1. `rag/solvers.py` with `tests/test_solvers.py` (known-answer checks)
2. `rag/ingest.py` chunking, tagging, embedding
3. `rag/retrieve.py` classification, hybrid search, diagram lookup, out-of-scope gate
4. `tools/make_diagrams.py` 21 assets and the ID registry
5. `rag/answer.py` Deep Agent with the solvers as tools
6. `data/testset.json` 34 labelled queries
7. `rag/evaluate.py` nine metrics, `rag/report.py` seven-section report, `cli.py`

---

## 4. Testing: what was found and what was changed

### 4.1 Solvers

All nine passed textbook known-answer checks on the first run: Master Theorem
cases 1-3 and the boundary case, CLRS matrix chain (15125), CLRS Bellman-Ford
figure 24.4, classic knapsack instances, LCS of ABCBDAB/BDCABA.

### 4.2 Ingestion defects

| Finding | Root cause | Fix |
|---|---|---|
| Table of contents parsed as real sections, producing 45-character chunks | Each heading appears twice in the notes | Keep only the last occurrence of each section number |
| "Minimum Spanning Tree" text attributed to the 0/1 knapsack section | Sub-headings are unnumbered, so the section regex missed them | Split section bodies on bare title lines as well |
| "Graph Colouring" chunk tagged as the tries topic | Naive substring matching: "Colours are **trie**d in order" | Word-boundary keyword matching with lookarounds |
| Section "4.3 0/1 Knapsack Problem" never detected, so `knapsack_01` had no chunks | The heading regex required the title to start with a letter, but this one starts with "0" | Accept an alphanumeric first character |
| Heaps chunk tagged as MST because it mentions Prim's in passing | A window's own keywords can point away from its section | Section-level topic prior: weak chunk evidence defers to the topic computed over the whole section |
| `job_sequencing` had no chunks | The topic appears only in the notes' table of contents, never in the body | Authored a job sequencing section in the supplement |

After these fixes all 29 syllabus topics have at least one chunk, and the only
untagged chunk is the textbook-list page of the syllabus PDF, which is correct.

### 4.3 Diagram rendering defects

Reading the generated PNGs revealed that column labels overlapped the first table
row, and that the LCS direction arrows required by the brief were computed but
never passed to the renderer. Both fixed; the LCS figure now matches CLRS
figure 15.8 with the traceback path highlighted.

### 4.4 First full evaluation run

Scores: relevance 0.567, recall 0.733, precision 0.933, faithfulness 0.700,
answer relevance 0.833, semantic accuracy 0.958, disambiguation 0.875,
diagram 1.000, robustness 0.971. Four root causes behind the misses:

1. **Q04 produced no answer.** The solver ran and returned 1085 characters, but
   the answer was empty. Cause: the text extractor accepted only assistant
   messages with no tool calls, so a message carrying both text and a tool call
   was discarded. Fixed, plus one retry when the answer comes back empty, plus
   support for providers that return content blocks rather than a string.

2. **Faithfulness was being under-scored on every numeric query.** The judge saw
   only the retrieved passages, so any number that came from a solver looked
   unsupported. This was a harness defect, not a system defect. Verified solver
   output is now included in the judge's context and labelled as trusted.

3. **Q15 routed correctly but ranked a general chunk first.** The topic filter
   needed three matching chunks before it would apply, and `matrix_chain` has
   three chunks in total, so it fell back to a unit filter that cannot separate
   topics inside unit 4. Replaced the hard filter with priority ranking:
   on-topic chunks always lead, remaining slots are backfilled by fused score.

4. **Q34 (CAP theorem) was answered instead of refused.** The keyword router
   matched "partition" from Quick Sort's keyword list and reported 0.95
   confidence, which suppressed the out-of-scope gate. Cause: confidence was
   computed purely from the margin over the runner-up, so a single weak keyword
   with no competitor looked certain. Confidence is now scaled by absolute
   evidence as well; the same query now scores 0.237 and is refused.

Separately, the judge flagged the AVL answer as thin on balance-factor detail.
That was a genuine corpus gap, so the supplement gained worked LL and LR examples
with explicit balance factors before and after rotation.

### 4.5 Second full evaluation run

Disambiguation, diagram linkage and robustness all reached 1.000 and precision
0.967. Two new findings:

5. **Q12 crashed with `KeyError: 'E'`.** The model passed a graph whose edge
   referenced a vertex with no adjacency entry of its own. The solvers assumed
   every referenced vertex was a key. Fixed by normalising the graph and
   validating shapes, and separately by wrapping every solver so an exception is
   returned to the model as an error rather than aborting the query.

6. **Context relevance fell from 0.567 to 0.500.** Priority ranking always
   returned five chunks, so a topic with only one or two chunks had its context
   padded with unrelated passages. The brief scores exactly this. Context size is
   now adaptive: take the on-topic chunks, and top up only as far as the three
   chunks the generator needs.

Also corrected a labelling error of our own: Q23 asks the student to *differentiate*
backtracking from branch and bound, so passages from either topic are pertinent.
Scoring now accepts a set of topics for such comparison questions.

### 4.6 Third full evaluation run

Relevance rose to 0.711 and precision to 1.000, but all three judge-scored metrics
came back null. Cause: appending full solver tables to the judge prompt pushed it
long enough that the model returned no JSON at all, and the judge had no retry, so
30 of 34 gradings silently voided. Fixed by clipping the judge's context and
answer, raising its token budget, retrying once, and recording the raw response
when parsing still fails.

### 4.7 Fourth full evaluation run

Recall, precision, faithfulness, answer relevance, semantic accuracy, diagram
linkage and robustness all reached 1.000, and relevance rose to 0.702. Two
problems remained.

7. **26 of 30 judge calls returned an empty completion.** The four scores that
   did land were all 1.000, which made the generation metrics look perfect while
   resting on a sample of four. Running the same gradings at lower concurrency
   returned content every time, so the endpoint was shedding load rather than
   refusing the prompt. Reduced the evaluation to two workers and gave both the
   judge and the query router three attempts with backoff. Both now record the
   raw response when they still fail, so an empty completion can never again be
   mistaken for a passing score.

8. **Q14 routed to no topic at all.** The word "knapsack" on its own appeared in
   no topic's keyword list: `fractional_knapsack` held "fractional knapsack" and
   `knapsack_01` held "0/1 knapsack", so a question phrased as "the knapsack
   problem ... items cannot be broken" matched nothing. Added the bare term to
   both topics, which deliberately leaves them tied, plus the phrases that
   actually separate them ("cannot be broken", "taken whole" against "can be
   broken", "fraction of an item").

### 4.8 Hardening the offline path

`tests/test_retrieve.py` exercises the deterministic keyword router with the
model switched off. It initially failed three checks, each a genuine weakness
rather than an unfair test:

- "Which dynamic programming algorithm finds the best order to multiply a chain
  of matrices" tied at one keyword each between `dp_principles` and
  `matrix_chain`, and the tie broke on declaration order. Promoted "chain of
  matrices" to a strong keyword and added the phrasings around it.
- The CAP theorem query still escaped the out-of-scope gate without the model,
  because the gate only fired when the topic was null, and the weak
  `sorting_comparison` match was not null. The gate now also fires on a topic
  held with low confidence.

All eight retrieval checks and all nine solver checks now pass with no API key.

### 4.9 Report structure

The generated answers carry their own `##` headings. Inserted verbatim into the
numerical demonstration section they registered as top-level report sections and
broke the document outline. Embedded answers are now demoted four heading levels.

---

## 5. Commands

```bash
python cli.py ingest                 # rebuild chunks + embeddings
python cli.py diagrams               # regenerate the 21 assets and registry
python cli.py ask "your question"
python cli.py eval                   # run all 34 labelled queries
python cli.py report                 # build out/report.md and out/report.pdf
python tests/test_solvers.py         # solver known-answer checks
```
