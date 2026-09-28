"""Retrieval checks. Run: python tests/test_retrieve.py

The deterministic keyword router and the hybrid ranker are tested without any
API key. The model-backed router is skipped when no key is configured.
"""
import json, os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rag.retrieve import lexical_classify, search, find_diagrams, retrieve, load_index

TESTSET = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "testset.json").read_text())
CASES = {c["id"]: c for c in TESTSET["queries"]}


def test_index_is_built():
    idx = load_index()
    assert len(idx["chunks"]) > 50, "index looks empty; run `python cli.py ingest`"
    assert len(idx["answerable"]) > 40, "no answerable chunks"

def test_every_syllabus_topic_has_a_chunk():
    idx = load_index()
    covered = {c["topic"] for c in idx["chunks"] if c["source_type"] != "pyq" and c["topic"]}
    syllabus = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "syllabus.json").read_text())
    expected = {t["topic"] for u in syllabus["units"] for t in u["topics"]}
    missing = expected - covered
    assert not missing, f"topics with no chunk: {sorted(missing)}"

def test_collision_pairs_route_apart():
    """Each colliding term must route its two queries to different topics."""
    for pair in TESTSET["collision_pairs"]:
        routed = []
        for qid in pair["queries"]:
            case = CASES[qid]
            cls = lexical_classify(case["query"])
            routed.append(cls["topic"])
            assert cls["topic"] == case["expected_topic"], (
                f"{qid} ('{pair['term']}') routed to {cls['topic']}, "
                f"expected {case['expected_topic']}")
        assert len(set(routed)) == len(routed), f"'{pair['term']}' collapsed to one topic"

def test_top1_chunk_is_on_topic_for_collisions():
    for pair in TESTSET["collision_pairs"]:
        for qid in pair["queries"]:
            case = CASES[qid]
            cls = lexical_classify(case["query"])
            chunks, _ = search(case["query"], cls)
            assert chunks, f"{qid} retrieved nothing"
            assert chunks[0].topic == case["expected_topic"], (
                f"{qid} top-1 is {chunks[0].topic}, expected {case['expected_topic']}")

def test_out_of_scope_queries_are_refused():
    for case in TESTSET["queries"]:
        if case["category"] != "oos":
            continue
        r = retrieve(case["query"], use_llm=False)
        assert r["refuse"], (
            f"{case['id']} not refused (max dense {r['debug']['max_dense']})")

def test_in_scope_queries_are_not_refused():
    for case in TESTSET["queries"]:
        if case["category"] == "oos":
            continue
        r = retrieve(case["query"], use_llm=False)
        assert not r["refuse"], f"{case['id']} wrongly refused: {r['refusal_reason']}"

def test_diagram_lookup_matches_expected_ids():
    for case in TESTSET["queries"]:
        want = case.get("expected_diagram_ids") or []
        if not want:
            continue
        cls = lexical_classify(case["query"])
        got = [d["id"] for d in find_diagrams(case["query"], cls)]
        assert got == want, f"{case['id']} matched {got}, expected {want}"

def test_diagram_registry_files_exist():
    root = pathlib.Path(__file__).resolve().parents[1]
    reg = json.loads((root / "data" / "diagrams" / "diagrams.json").read_text())
    ids = [d["id"] for d in reg["diagrams"]]
    assert len(ids) == len(set(ids)), "duplicate diagram id in the registry"
    for d in reg["diagrams"]:
        assert (root / "data" / "diagrams" / d["file"]).exists(), f"missing asset {d['file']}"
        assert d["id"].startswith(f"DIAG_DAA_U{d['unit']}_"), f"{d['id']} disagrees with its unit"

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    print("all retrieval checks passed" if not fails else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
