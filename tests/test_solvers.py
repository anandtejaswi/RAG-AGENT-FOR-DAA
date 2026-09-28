"""Known-answer checks for the numeric solvers. Run: python tests/test_solvers.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rag.solvers import (master_theorem, recursion_tree, knapsack_01, fractional_knapsack,
                         lcs, matrix_chain, floyd_warshall, dijkstra, bellman_ford)

def test_master_theorem():
    r = master_theorem(2, 2, 1)                      # merge sort
    assert r["case"] == 2 and r["bound"] == "Theta(n log n)", r
    r = master_theorem(9, 3, 1)                      # CLRS case 1
    assert r["case"] == 1 and r["bound"] == "Theta(n^2)", r
    r = master_theorem(3, 4, 1)                      # CLRS case 3
    assert r["case"] == 3 and r["bound"] == "Theta(n)", r
    r = master_theorem(1, 2, 0)                      # binary search
    assert r["case"] == 2 and r["bound"] == "Theta(1 log n)", r
    r = master_theorem(7, 2, 2)                      # Strassen
    assert r["case"] == 1 and "log_2(7)" in r["bound"], r
    r = master_theorem(4, 2, 2)                      # boundary, case 2
    assert r["case"] == 2 and r["bound"] == "Theta(n^2 log n)", r

def test_recursion_tree():
    r = recursion_tree(2, 2, 1, levels=4)
    assert r["total"] == "Theta(n log n)" and len(r["levels"]) == 4, r
    r = recursion_tree(4, 2, 1, levels=3)
    assert "log_2(4)" in r["total"], r

def test_knapsack_01():
    r = knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert r["optimal_value"] == 9 and r["chosen_items"] == [2, 3], r
    r = knapsack_01([10, 20, 30], [60, 100, 120], 50)
    assert r["optimal_value"] == 220 and r["chosen_items"] == [2, 3], r
    assert len(r["table"]) == 4 and len(r["table"][0]) == 51

def test_fractional_knapsack():
    r = fractional_knapsack([10, 20, 30], [60, 100, 120], 50)
    assert r["optimal_value"] == 240.0, r
    assert r["greedy_order"] == [1, 2, 3], r

def test_lcs():
    r = lcs("ABCBDAB", "BDCABA")
    assert r["length"] == 4 and len(r["lcs"]) == 4, r
    assert all(ch in "ABCBDAB" for ch in r["lcs"])
    r = lcs("AGGTAB", "GXTXAYB")
    assert r["length"] == 4 and r["lcs"] == "GTAB", r

def test_matrix_chain():
    r = matrix_chain([30, 35, 15, 5, 10, 20, 25])   # CLRS
    assert r["min_cost"] == 15125, r
    assert r["parenthesization"] == "((A1(A2A3))((A4A5)A6))", r
    r = matrix_chain([10, 20, 30])
    assert r["min_cost"] == 6000, r

def test_floyd_warshall():
    W = [[0, 3, None, 7], [8, 0, 2, None], [5, None, 0, 1], [2, None, None, 0]]
    r = floyd_warshall(W)
    assert r["final"] == [[0, 3, 5, 6], [5, 0, 2, 3], [3, 6, 0, 1], [2, 5, 7, 0]], r["final"]
    assert len(r["stages"]) == 5 and not r["negative_cycle"]

def test_dijkstra():
    g = {"A": {"B": 4, "C": 2}, "B": {"C": 5, "D": 10}, "C": {"E": 3},
         "D": {"F": 11}, "E": {"D": 4}, "F": {}}
    r = dijkstra(g, "A")
    assert r["distances"] == {"A": 0, "B": 4, "C": 2, "D": 9, "E": 5, "F": 20}, r["distances"]
    assert [t["extracted"] for t in r["trace"]] == ["A", "C", "B", "E", "D", "F"], r["trace"]
    try:
        dijkstra({"A": {"B": -1}, "B": {}}, "A"); assert False, "negative weight not rejected"
    except ValueError:
        pass

def test_bellman_ford():
    g = {"s": {"t": 6, "y": 7}, "t": {"x": 5, "y": 8, "z": -4},
         "x": {"t": -2}, "y": {"x": -3, "z": 9}, "z": {"s": 2, "x": 7}}
    r = bellman_ford(g, "s")                          # CLRS figure 24.4
    assert r["distances"] == {"s": 0, "t": 2, "x": 4, "y": 7, "z": -2}, r["distances"]
    assert not r["negative_cycle"]
    g2 = {"a": {"b": 1}, "b": {"c": -4}, "c": {"a": 1}}
    assert bellman_ford(g2, "a")["negative_cycle"], "negative cycle missed"

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    print("all solver checks passed" if not fails else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
