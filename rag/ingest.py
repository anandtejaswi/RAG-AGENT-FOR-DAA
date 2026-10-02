"""Corpus ingestion: sources -> tagged chunks -> dense index.

Chunk boundaries follow the headings of the study material, because a DAA
section is a self-contained explanation. Long sections are then windowed with
LangChain's RecursiveCharacterTextSplitter. Each chunk is tagged with the
syllabus unit and topic it belongs to, which is what makes cross-topic
disambiguation possible at retrieval time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX = ROOT / "index"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Headings in the supplied notes look like " 1.2 Recurrences & Master Theorem".
# The title may start with a digit ("4.3 0/1 Knapsack Problem"), so the first
# character is alphanumeric rather than an upper-case letter.
SECTION_RE = re.compile(r"^\s*(\d+\.\d+[a-z]?)\s+([A-Za-z0-9][^\n]{3,80})$", re.M)
MD_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.M)
PAGE_RE = re.compile(r"^\s*Page \d+\s*$", re.M)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    source_type: str
    heading: str
    unit: int | None
    topic: str | None
    topic_score: float
    page: int | None = None


def load_syllabus() -> dict:
    return json.loads((DATA / "syllabus.json").read_text())


def topic_table(syllabus: dict) -> list[dict]:
    rows = []
    for unit in syllabus["units"]:
        for t in unit["topics"]:
            rows.append(
                {
                    "unit": unit["unit"],
                    "unit_name": unit["name"],
                    "topic": t["topic"],
                    "title": t["title"],
                    "keywords": [k.lower() for k in t["keywords"]],
                    "strong": [k.lower() for k in t.get("strong", [])],
                }
            )
    return rows


_KW_CACHE: dict[str, re.Pattern] = {}


def _kw_re(kw: str) -> re.Pattern:
    """Word-boundary matcher.

    Plain substring counting is wrong here: 'tried' contains 'trie' and
    'relaxation' style short keys collide across topics. Lookarounds are used
    instead of \\b so keywords ending in punctuation, such as 't(n)', still
    anchor correctly.
    """
    pat = _KW_CACHE.get(kw)
    if pat is None:
        # A trailing 's' is optional so 'binomial heap' also matches 'heaps'.
        pat = re.compile(r"(?<!\w)" + re.escape(kw) + r"s?(?!\w)")
        _KW_CACHE[kw] = pat
    return pat


def tag_topic(text: str, heading: str, topics: list[dict]) -> tuple[int | None, str | None, float]:
    """Score every syllabus topic against a chunk and return the best match.

    Heading matches count triple because a section heading names its topic
    directly, and 'strong' keywords count triple again because they are the
    terms that separate colliding topics.
    """
    hay = text.lower()
    head = heading.lower()
    best, best_score = None, 0.0
    for t in topics:
        score = 0.0
        for kw in t["keywords"]:
            n = len(_kw_re(kw).findall(hay))
            if n:
                score += min(n, 4)
            if _kw_re(kw).search(head):
                score += 3
        for kw in t["strong"]:
            n = len(_kw_re(kw).findall(hay))
            if n:
                score += 3 * min(n, 3)
            if _kw_re(kw).search(head):
                score += 9
        if score > best_score:
            best, best_score = t, score
    if best is None:
        return None, None, 0.0
    return best["unit"], best["topic"], round(best_score, 2)


SUBHEAD_RE = re.compile(
    r"^(?!\s*[-*•])[ \t]*([A-Z][A-Za-z0-9 '()\-/&,.]{2,55})[ \t]*$\n\s*\n", re.M
)


def split_subsections(heading: str, body: str) -> list[tuple[str, str]]:
    """Split a section body on unnumbered sub-headings.

    The supplied notes label sub-topics with a bare title line ('Merge Sort',
    'Minimum Spanning Tree (MST)'). Without this split, MST text inherits the
    heading of the previous numbered section and is mis-attributed.
    """
    marks = list(SUBHEAD_RE.finditer(body))
    if not marks:
        return [(heading, body)]
    out = []
    if body[: marks[0].start()].strip():
        out.append((heading, body[: marks[0].start()].strip()))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        sub = m.group(1).strip()
        out.append((f"{heading} > {sub}", body[m.end(): end].strip()))
    return out


def read_text_notes(path: Path) -> list[tuple[str, str]]:
    """Split the plain-text notes into (heading, body) sections."""
    raw = PAGE_RE.sub("", path.read_text(encoding="utf-8", errors="ignore"))
    matches = list(SECTION_RE.finditer(raw))
    if not matches:
        return [("", raw)]

    # Each heading appears twice: once in the table of contents and once above
    # the real body. Keep the last occurrence of each section number.
    last: dict[str, int] = {}
    for i, m in enumerate(matches):
        last[m.group(1)] = i

    sections = []
    for i, m in enumerate(matches):
        if last[m.group(1)] != i:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        heading = f"{m.group(1)} {m.group(2).strip()}"
        sections.extend(split_subsections(heading, raw[start:end].strip()))
    return sections


def read_markdown(path: Path) -> list[tuple[str, str]]:
    raw = path.read_text(encoding="utf-8")
    matches = list(MD_HEADING_RE.finditer(raw))
    if not matches:
        return [("", raw)]
    sections = []
    parent = ""
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        title = m.group(2).strip()
        if len(m.group(1)) == 2:          # '##' starts a new parent section
            parent = title
            heading = title
        else:                              # '###' hangs off the current parent
            heading = f"{parent} > {title}" if parent else title
        sections.append((heading, raw[start:end].strip()))
    return sections


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    import pymupdf

    out = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                out.append((i, text))
    return out


def splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def build_chunks() -> list[Chunk]:
    syllabus = load_syllabus()
    topics = topic_table(syllabus)
    split = splitter()
    chunks: list[Chunk] = []
    counter = 0

    # A chunk is a window into a section, and a section is about one topic. When
    # a window's own keyword evidence is weak or points elsewhere (a heaps
    # section that mentions Prim's MST in passing), the topic computed over the
    # whole section wins.
    SECTION_PRIOR_MARGIN = 12.0

    def add(text: str, source: str, source_type: str, heading: str,
            page: int | None = None, section_topic: tuple | None = None):
        nonlocal counter
        text = text.strip()
        if len(text) < 40:
            return
        unit, topic, score = tag_topic(text, heading, topics)
        if section_topic and section_topic[1]:
            s_unit, s_topic, s_score = section_topic
            if s_topic != topic and score < SECTION_PRIOR_MARGIN and s_score > score:
                unit, topic, score = s_unit, s_topic, score
        counter += 1
        chunks.append(
            Chunk(
                id=f"C{counter:04d}",
                text=text,
                source=source,
                source_type=source_type,
                heading=heading,
                unit=unit,
                topic=topic,
                topic_score=score,
                page=page,
            )
        )

    # 1. Supplied study notes.
    notes = DATA / "1038678511-Daa-Aktu-Notes.txt"
    if notes.exists():
        for heading, body in read_text_notes(notes):
            prior = tag_topic(body, heading, topics)
            for piece in split.split_text(body):
                add(piece, notes.name, "notes", heading, section_topic=prior)

    # 2. Authored supplementary notes covering syllabus gaps.
    supp = DATA / "notes_supplement.md"
    if supp.exists():
        for heading, body in read_markdown(supp):
            prior = tag_topic(body, heading, topics)
            for piece in split.split_text(body):
                add(piece, supp.name, "notes_supplement", heading, section_topic=prior)

    # 3. Official syllabus document.
    syl_pdf = DATA / "syllabus-daa"
    if syl_pdf.exists():
        for page_no, text in read_pdf_pages(syl_pdf):
            for piece in split.split_text(text):
                add(piece, "syllabus-daa.pdf", "syllabus", "KCS-503 detailed syllabus", page_no)

    # 4. Previous year question papers, indexed but excluded from answer
    #    context (they contain questions, not explanations).
    pyq_dir = DATA / "pyq"
    if pyq_dir.is_dir():
        for pdf in sorted(pyq_dir.glob("*.pdf")):
            year = pdf.stem
            for page_no, text in read_pdf_pages(pdf):
                for piece in split.split_text(text):
                    add(piece, f"pyq/{pdf.name}", "pyq", f"AKTU DAA question paper {year}", page_no)

    return chunks


def embed_texts(texts: list[str], model=None) -> np.ndarray:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    if model is None:
        try:
            model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except Exception:
            model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(
        texts, batch_size=16, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    return vecs.astype(np.float32)


def main() -> None:
    INDEX.mkdir(exist_ok=True)
    chunks = build_chunks()
    print(f"built {len(chunks)} chunks")

    by_type: dict[str, int] = {}
    untagged = 0
    for c in chunks:
        by_type[c.source_type] = by_type.get(c.source_type, 0) + 1
        if c.topic is None:
            untagged += 1
    print("by source:", by_type)
    print(f"untagged chunks: {untagged}")

    with (INDEX / "chunks.jsonl").open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")

    vecs = embed_texts([f"{c.heading}\n{c.text}" for c in chunks])
    np.save(INDEX / "embeddings.npy", vecs)
    (INDEX / "meta.json").write_text(
        json.dumps({"model": EMBED_MODEL, "count": len(chunks), "dim": int(vecs.shape[1])}, indent=2)
    )
    print(f"embedded {vecs.shape[0]} chunks, dim {vecs.shape[1]}")


if __name__ == "__main__":
    main()
