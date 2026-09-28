"""Generate the diagram assets and their persistent-ID registry.

Every figure that shows a computed table or trace is rendered from the solvers
in rag.solvers, so a diagram and the answer text can never disagree.

Run: python tools/make_diagrams.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "diagrams"

from rag.solvers import (knapsack_01, lcs, matrix_chain, floyd_warshall,  # noqa: E402
                         dijkstra, bellman_ford)

REGISTRY: list[dict] = []
INK = "#1a1a1a"
ACCENT = "#0b6cb0"
WARM = "#c0392b"
MUTED = "#8a8a8a"


def register(diag_id, unit, topic, concept, keywords, caption, fig):
    OUT.mkdir(parents=True, exist_ok=True)
    fname = diag_id.lower().replace("diag_daa_", "") + ".png"
    fig.savefig(OUT / fname, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    REGISTRY.append(
        {
            "id": diag_id,
            "unit": unit,
            "topic": topic,
            "concept": concept,
            "keywords": keywords,
            "file": fname,
            "caption": caption,
        }
    )
    print(f"  {diag_id} -> {fname}")


def new_fig(w=7.2, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    return fig, ax


def draw_table(ax, rows, col_labels=None, row_labels=None, title="",
               highlight=None, arrows=None, cell_w=0.8, cell_h=0.52, x0=0.0, y0=0.0):
    """Draw a DP table; `highlight` is a set of (i, j) cells, `arrows` a dict."""
    highlight = highlight or set()
    arrows = arrows or {}
    n_rows, n_cols = len(rows), len(rows[0])
    for i in range(n_rows):
        for j in range(n_cols):
            x, y = x0 + j * cell_w, y0 - i * cell_h
            face = "#ffeaa7" if (i, j) in highlight else "white"
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, fill=True,
                                   facecolor=face, edgecolor=MUTED, lw=0.8))
            val = rows[i][j]
            txt = "∞" if val is None else str(val)
            arrow = arrows.get((i, j), "")
            ax.text(x + cell_w / 2, y + cell_h / 2, f"{arrow}{txt}",
                    ha="center", va="center", fontsize=8.5, color=INK)
    if col_labels:
        for j, lab in enumerate(col_labels):
            ax.text(x0 + j * cell_w + cell_w / 2, y0 + cell_h * 1.35, str(lab),
                    ha="center", va="center", fontsize=8.5, color=ACCENT, weight="bold")
    if row_labels:
        for i, lab in enumerate(row_labels):
            ax.text(x0 - 0.12, y0 - i * cell_h + cell_h / 2, str(lab),
                    ha="right", va="center", fontsize=8.5, color=ACCENT, weight="bold")
    if title:
        ax.text(x0, y0 + cell_h * 1.7, title, fontsize=10.5, weight="bold", color=INK)
    ax.set_xlim(x0 - 1.6, x0 + n_cols * cell_w + 0.3)
    ax.set_ylim(y0 - n_rows * cell_h - 0.3, y0 + cell_h * 2.8)


def draw_graph(ax, pos, edges, directed=False, edge_labels=True,
               node_colors=None, edge_colors=None, title="", radius=0.26):
    node_colors = node_colors or {}
    edge_colors = edge_colors or {}
    for (u, v, w) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        col = edge_colors.get((u, v), edge_colors.get((v, u), MUTED))
        lw = 2.4 if col != MUTED else 1.1
        style = "-|>" if directed else "-"
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=14, color=col, lw=lw,
                                     shrinkA=radius * 72 * 0.42, shrinkB=radius * 72 * 0.42))
        if edge_labels:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.12, str(w), fontsize=8.5, color=col,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), radius, facecolor=node_colors.get(name, "white"),
                            edgecolor=INK, lw=1.4, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5,
                weight="bold", color=INK, zorder=4)
    if title:
        ax.set_title(title, fontsize=10.5, weight="bold", color=INK)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_aspect("equal")


def draw_tree(ax, nodes, edges, title="", radius=0.26, colors=None):
    colors = colors or {}
    for u, v in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.2, zorder=1)
    for name, (x, y) in nodes.items():
        label = name.split("#")[0]
        ax.add_patch(Circle((x, y), radius, facecolor=colors.get(name, "white"),
                            edgecolor=INK, lw=1.4, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                weight="bold", color="white" if colors.get(name) in (INK, WARM) else INK, zorder=4)
    if title:
        ax.set_title(title, fontsize=10.5, weight="bold", color=INK)
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal")


# --------------------------------------------------------------------------
# Unit 1
# --------------------------------------------------------------------------

def recursion_tree_diagram():
    fig, ax = new_fig(7.4, 4.6)
    levels = [(0, ["n"]), (1, ["n/2", "n/2"]), (2, ["n/4"] * 4), (3, ["..."] * 8)]
    span = 6.0
    for depth, labels in levels:
        y = -depth * 1.1
        k = len(labels)
        for i, lab in enumerate(labels):
            x = (i + 0.5) * span / k - span / 2
            ax.add_patch(Circle((x, y), 0.22, facecolor="white", edgecolor=INK, lw=1.3, zorder=3))
            ax.text(x, y, lab, ha="center", va="center", fontsize=7.5, zorder=4)
            if depth:
                px = ((i // 2) + 0.5) * span / (k // 2) - span / 2
                ax.plot([px, x], [y + 1.1, y], color=MUTED, lw=1.0, zorder=1)
        cost = ["n", "n", "n", "n"][depth]
        ax.text(span / 2 + 0.55, y, f"cost = {cost}", fontsize=9, color=ACCENT,
                va="center", weight="bold")
    ax.text(0, -4.0, "log₂n + 1 levels, each costing n  →  T(n) = Θ(n log n)",
            ha="center", fontsize=10, color=WARM, weight="bold")
    ax.set_title("Recursion tree for T(n) = 2T(n/2) + n", fontsize=11, weight="bold")
    ax.set_xlim(-3.6, 4.6)
    ax.set_ylim(-4.4, 0.6)
    ax.set_axis_off()
    register("DIAG_DAA_U1_RECUR_TREE_01", 1, "recurrences", "recursion tree for T(n)=2T(n/2)+n",
             ["recursion tree", "recurrence tree", "t(n)=2t(n/2)+n", "merge sort recurrence"],
             "Recursion tree expansion showing log n levels of cost n each, giving Theta(n log n).",
             fig)


def master_cases_diagram():
    fig, ax = new_fig(7.6, 3.9)
    boxes = [
        (0.5, 2.6, "Compare f(n) with n^log_b(a)", "#eaf3fb"),
        (-2.2, 1.2, "f(n) = O(n^(log_b a - ε))\nCase 1", "#e8f6ec"),
        (0.5, 1.2, "f(n) = Θ(n^log_b a)\nCase 2", "#fff4e0"),
        (3.2, 1.2, "f(n) = Ω(n^(log_b a + ε))\nCase 3", "#fdeaea"),
        (-2.2, -0.2, "T(n) = Θ(n^log_b a)", "white"),
        (0.5, -0.2, "T(n) = Θ(n^log_b a · log n)", "white"),
        (3.2, -0.2, "T(n) = Θ(f(n))\nif a f(n/b) ≤ c f(n)", "white"),
    ]
    for x, y, text, colour in boxes:
        ax.add_patch(Rectangle((x - 1.15, y - 0.36), 2.3, 0.72, facecolor=colour,
                               edgecolor=INK, lw=1.1, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, zorder=3)
    for x in (-2.2, 0.5, 3.2):
        ax.add_patch(FancyArrowPatch((0.5, 2.24), (x, 1.58), arrowstyle="-|>",
                                     mutation_scale=12, color=MUTED, lw=1.1))
        ax.add_patch(FancyArrowPatch((x, 0.82), (x, 0.18), arrowstyle="-|>",
                                     mutation_scale=12, color=MUTED, lw=1.1))
    ax.set_title("Master Theorem: case selection for T(n) = aT(n/b) + f(n)",
                 fontsize=11, weight="bold")
    ax.set_xlim(-3.6, 4.6)
    ax.set_ylim(-0.9, 3.3)
    register("DIAG_DAA_U1_MASTER_CASES_01", 1, "recurrences", "Master Theorem three cases flowchart",
             ["master theorem", "master method", "three cases", "case 1", "case 2", "case 3"],
             "Decision chart for the three cases of the Master Theorem and the bound each yields.",
             fig)


def merge_sort_diagram():
    fig, ax = new_fig(7.4, 4.0)
    divide = [["38", "27", "43", "3", "9", "82", "10"], ["38 27 43 3", "9 82 10"],
              ["38 27", "43 3", "9 82", "10"], ["38", "27", "43", "3", "9", "82", "10"]]
    for d, row in enumerate(divide):
        y = -d * 0.85
        k = len(row)
        for i, lab in enumerate(row):
            x = (i + 0.5) * 7.0 / k - 3.5
            ax.add_patch(Rectangle((x - 0.42, y - 0.2), 0.84, 0.4, facecolor="#eaf3fb",
                                   edgecolor=INK, lw=1.0))
            ax.text(x, y, lab, ha="center", va="center", fontsize=7.5)
    merged = ["3 9 10 27 38 43 82"]
    ax.add_patch(Rectangle((-1.6, -3.9), 3.2, 0.44, facecolor="#e8f6ec", edgecolor=INK, lw=1.2))
    ax.text(0, -3.68, merged[0], ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(-3.5, 0.55, "divide", fontsize=9.5, color=ACCENT, weight="bold")
    ax.text(-3.5, -3.68, "merge", fontsize=9.5, color=WARM, weight="bold")
    ax.annotate("", xy=(0, -3.45), xytext=(0, -2.75),
                arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.6))
    ax.set_title("Merge sort: divide to single elements, then merge (T(n)=2T(n/2)+Θ(n))",
                 fontsize=10.5, weight="bold")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.2, 0.9)
    register("DIAG_DAA_U1_MERGE_SORT_01", 1, "sorting_comparison", "merge sort divide and merge stages",
             ["merge sort", "divide and merge", "merge step"],
             "Merge sort splitting an array to singletons and merging back in sorted order.", fig)


# --------------------------------------------------------------------------
# Unit 2
# --------------------------------------------------------------------------

def avl_ll_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for a in (ax1, ax2):
        a.set_axis_off()
    before = {"30#a": (0, 1.0), "20#a": (-0.8, 0.2), "10#a": (-1.6, -0.6)}
    draw_tree(ax1, before, [("30#a", "20#a"), ("20#a", "10#a")],
              title="Before: balance factor of 30 is +2 (LL case)",
              colors={"30#a": WARM})
    after = {"20#b": (0, 1.0), "10#b": (-0.8, 0.2), "30#b": (0.8, 0.2)}
    draw_tree(ax2, after, [("20#b", "10#b"), ("20#b", "30#b")],
              title="After: single right rotation about 30", colors={"20#b": "#e8f6ec"})
    fig.suptitle("AVL LL imbalance repaired by one right rotation", fontsize=11, weight="bold")
    register("DIAG_DAA_U2_AVL_ROT_01", 2, "red_black_trees", "AVL LL single right rotation",
             ["avl", "ll rotation", "single rotation", "right rotation", "balance factor"],
             "An LL imbalance and the single right rotation that restores the AVL property.", fig)


def avl_lr_diagram():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for a in axes:
        a.set_axis_off()
    t1 = {"30#c": (0, 1.0), "10#c": (-0.9, 0.2), "20#c": (-0.2, -0.6)}
    draw_tree(axes[0], t1, [("30#c", "10#c"), ("10#c", "20#c")],
              title="LR case at 30", colors={"30#c": WARM})
    t2 = {"30#d": (0, 1.0), "20#d": (-0.9, 0.2), "10#d": (-1.7, -0.6)}
    draw_tree(axes[1], t2, [("30#d", "20#d"), ("20#d", "10#d")],
              title="Step 1: left rotation at 10")
    t3 = {"20#e": (0, 1.0), "10#e": (-0.8, 0.2), "30#e": (0.8, 0.2)}
    draw_tree(axes[2], t3, [("20#e", "10#e"), ("20#e", "30#e")],
              title="Step 2: right rotation at 30", colors={"20#e": "#e8f6ec"})
    fig.suptitle("AVL LR imbalance repaired by a double rotation", fontsize=11, weight="bold")
    register("DIAG_DAA_U2_AVL_ROT_02", 2, "red_black_trees", "AVL LR double rotation",
             ["avl", "lr rotation", "double rotation", "left right rotation"],
             "An LR imbalance repaired by a left rotation followed by a right rotation.", fig)


def rb_insert_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for a in (ax1, ax2):
        a.set_axis_off()
    before = {"11#r": (0, 1.0), "2#r": (-1.0, 0.2), "14#r": (1.0, 0.2),
              "1#r": (-1.7, -0.6), "7#r": (-0.3, -0.6)}
    draw_tree(ax1, before, [("11#r", "2#r"), ("11#r", "14#r"), ("2#r", "1#r"), ("2#r", "7#r")],
              title="Red parent and red uncle after inserting 7",
              colors={"11#r": INK, "2#r": WARM, "14#r": WARM, "1#r": INK, "7#r": WARM})
    after = {"11#s": (0, 1.0), "2#s": (-1.0, 0.2), "14#s": (1.0, 0.2),
             "1#s": (-1.7, -0.6), "7#s": (-0.3, -0.6)}
    draw_tree(ax2, after, [("11#s", "2#s"), ("11#s", "14#s"), ("2#s", "1#s"), ("2#s", "7#s")],
              title="Case 1: recolour parent, uncle and grandparent",
              colors={"11#s": INK, "2#s": INK, "14#s": INK, "1#s": WARM, "7#s": WARM})
    fig.suptitle("Red-Black insertion fix-up, case 1 (red uncle → recolour)",
                 fontsize=11, weight="bold")
    register("DIAG_DAA_U2_RB_INSERT_01", 2, "red_black_trees", "red-black tree insertion rebalancing",
             ["red-black", "red black tree", "recolour", "recoloring", "rebalancing", "fix-up"],
             "Red-Black insertion fix-up when the uncle is red: recolour instead of rotating.", fig)


def btree_split_diagram():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 3.8))
    for a in (ax1, ax2):
        a.set_axis_off()

    def node(ax, x, y, keys, w=0.52):
        for i, k in enumerate(keys):
            ax.add_patch(Rectangle((x + i * w, y), w, 0.42, facecolor="white",
                                   edgecolor=INK, lw=1.2))
            ax.text(x + i * w + w / 2, y + 0.21, str(k), ha="center", va="center", fontsize=9)

    node(ax1, -1.3, 0, ["A", "D", "F", "H", "L", "N", "P"])
    ax1.set_title("Full node of a B-tree with t = 4 (2t−1 = 7 keys)", fontsize=10, weight="bold")
    ax1.set_xlim(-1.8, 2.6)
    ax1.set_ylim(-0.2, 0.8)
    node(ax2, 0.2, 0.75, ["H"])
    node(ax2, -1.3, 0.0, ["A", "D", "F"])
    node(ax2, 1.1, 0.0, ["L", "N", "P"])
    ax2.plot([0.35, 0.05], [0.75, 0.42], color=MUTED, lw=1.0)
    ax2.plot([0.6, 1.9], [0.75, 0.42], color=MUTED, lw=1.0)
    ax2.set_title("After split: median H rises to the parent", fontsize=10, weight="bold")
    ax2.set_xlim(-1.8, 2.9)
    ax2.set_ylim(-0.2, 1.4)
    register("DIAG_DAA_U2_BTREE_SPLIT_01", 2, "b_trees", "B-tree node split on the median key",
             ["b-tree", "split child", "node split", "median key", "minimum degree"],
             "Splitting a full B-tree node: the median key moves up into the parent.", fig)


def binomial_heap_diagram():
    fig, ax = new_fig(7.6, 3.2)
    layouts = {
        "B0": {"a": (0, 0)},
        "B1": {"a": (1.4, 0.5), "b": (1.4, -0.4)},
        "B2": {"a": (3.0, 0.9), "b": (2.6, 0.0), "c": (3.6, 0.0), "d": (2.6, -0.9)},
    }
    edges = {"B0": [], "B1": [("a", "b")], "B2": [("a", "b"), ("a", "c"), ("b", "d")]}
    for name, nodes in layouts.items():
        for u, v in edges[name]:
            ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color=MUTED, lw=1.1)
        for pos in nodes.values():
            ax.add_patch(Circle(pos, 0.19, facecolor="white", edgecolor=INK, lw=1.3, zorder=3))
        xs = [p[0] for p in nodes.values()]
        ax.text(sum(xs) / len(xs), 1.45, name, ha="center", fontsize=10,
                weight="bold", color=ACCENT)
        ax.text(sum(xs) / len(xs), -1.55, f"{2 ** int(name[1])} node(s)",
                ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("Binomial trees B₀, B₁, B₂: a binomial heap is a list of these",
                 fontsize=10.5, weight="bold")
    ax.set_xlim(-0.8, 4.4)
    ax.set_ylim(-1.9, 1.8)
    register("DIAG_DAA_U2_BINOMIAL_HEAP_01", 2, "binomial_fibonacci_heaps",
             "binomial trees B0 B1 B2 structure",
             ["binomial heap", "binomial tree", "b0", "b1", "b2", "mergeable heap"],
             "Binomial trees of order 0, 1 and 2, the building blocks of a binomial heap.", fig)


# --------------------------------------------------------------------------
# Unit 3
# --------------------------------------------------------------------------

GRAPH_POS = {"A": (0, 1.2), "B": (1.5, 1.9), "C": (1.5, 0.4), "D": (3.0, 1.9),
             "E": (3.0, 0.4), "F": (4.4, 1.2)}
GRAPH_EDGES = [("A", "B", 4), ("A", "C", 2), ("B", "C", 5), ("B", "D", 10),
               ("C", "E", 3), ("E", "D", 4), ("D", "F", 11)]


def bfs_dfs_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for a in (ax1, ax2):
        a.set_axis_off()
    bfs_order = ["A", "B", "C", "D", "E", "F"]
    dfs_order = ["A", "B", "C", "E", "D", "F"]
    draw_graph(ax1, GRAPH_POS, GRAPH_EDGES, edge_labels=False,
               node_colors={n: "#eaf3fb" for n in bfs_order},
               title="BFS from A (queue): " + " → ".join(bfs_order))
    draw_graph(ax2, GRAPH_POS, GRAPH_EDGES, edge_labels=False,
               node_colors={n: "#fff4e0" for n in dfs_order},
               title="DFS from A (stack): " + " → ".join(dfs_order))
    fig.suptitle("Graph traversals, both Θ(V + E)", fontsize=11, weight="bold")
    register("DIAG_DAA_U3_BFS_DFS_01", 3, "graph_traversal", "BFS and DFS visit order",
             ["bfs", "dfs", "breadth-first", "depth-first", "traversal order"],
             "Breadth-first and depth-first visit orders on the same graph.", fig)


def kruskal_diagram():
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for a in axes:
        a.set_axis_off()
    picks = [[("A", "C")], [("A", "C"), ("C", "E")], [("A", "C"), ("C", "E"), ("A", "B")]]
    caps = ["Pick A-C (w=2)", "Pick C-E (w=3)", "Pick A-B (w=4)"]
    for ax, chosen, cap in zip(axes, picks, caps):
        draw_graph(ax, GRAPH_POS, GRAPH_EDGES,
                   edge_colors={e: ACCENT for e in chosen}, title=cap)
    fig.suptitle("Kruskal's algorithm: add the lightest edge that creates no cycle",
                 fontsize=11, weight="bold")
    register("DIAG_DAA_U3_MST_KRUSKAL_01", 3, "mst", "Kruskal MST construction stages",
             ["kruskal", "minimum spanning tree", "mst stages", "lightest edge", "disjoint set"],
             "Successive stages of Kruskal's algorithm adding the lightest acyclic edges.", fig)


def prim_diagram():
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for a in axes:
        a.set_axis_off()
    grown = [[("A", "C")], [("A", "C"), ("C", "E")], [("A", "C"), ("C", "E"), ("A", "B")]]
    caps = ["Tree = {A}, add A-C", "Tree = {A,C}, add C-E", "Tree = {A,C,E}, add A-B"]
    for ax, chosen, cap in zip(axes, grown, caps):
        in_tree = {n for e in chosen for n in e} | {"A"}
        draw_graph(ax, GRAPH_POS, GRAPH_EDGES,
                   edge_colors={e: WARM for e in chosen},
                   node_colors={n: "#fdeaea" for n in in_tree}, title=cap)
    fig.suptitle("Prim's algorithm: grow one tree by its cheapest outgoing edge",
                 fontsize=11, weight="bold")
    register("DIAG_DAA_U3_MST_PRIM_01", 3, "mst", "Prim MST growth stages",
             ["prim", "minimum spanning tree", "grow tree", "cheapest edge", "cut"],
             "Prim's algorithm growing a single tree from vertex A, one cheapest edge at a time.",
             fig)


def dijkstra_diagram():
    graph = {"A": {"B": 4, "C": 2}, "B": {"C": 5, "D": 10}, "C": {"E": 3},
             "D": {"F": 11}, "E": {"D": 4}, "F": {}}
    res = dijkstra(graph, "A")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6),
                                   gridspec_kw={"width_ratios": [1.1, 1.0]})
    for a in (ax1, ax2):
        a.set_axis_off()
    tree = {(p, v) for v, p in res["predecessors"].items() if p}
    draw_graph(ax1, GRAPH_POS, GRAPH_EDGES, directed=True,
               edge_colors={e: ACCENT for e in tree},
               node_colors={n: "#eaf3fb" for n in graph},
               title="Shortest path tree from A")
    verts = list(graph)
    rows = [[t["extracted"]] + [("∞" if t["distances"][v] is None else int(t["distances"][v]))
                                for v in verts] for t in res["trace"]]
    draw_table(ax2, rows, col_labels=["extract"] + verts,
               row_labels=[f"it {t['iteration']}" for t in res["trace"]],
               title="Distance array after each extraction", cell_w=0.62, cell_h=0.44)
    fig.suptitle("Dijkstra trace: relax d[v] = min(d[v], d[u] + w(u,v))",
                 fontsize=11, weight="bold")
    register("DIAG_DAA_U3_DIJKSTRA_01", 3, "dijkstra", "Dijkstra distance array trace",
             ["dijkstra", "shortest path tree", "distance array", "trace", "extract-min"],
             "Dijkstra's shortest path tree from A with the distance array after each extraction.",
             fig)


def bellman_ford_diagram():
    graph = {"s": {"t": 6, "y": 7}, "t": {"x": 5, "y": 8, "z": -4},
             "x": {"t": -2}, "y": {"x": -3, "z": 9}, "z": {"s": 2, "x": 7}}
    res = bellman_ford(graph, "s")
    fig, ax = new_fig(7.0, 3.4)
    verts = list(graph)
    rows = [[("∞" if p["distances"][v] is None else int(p["distances"][v])) for v in verts]
            for p in res["passes"]]
    draw_table(ax, rows, col_labels=verts,
               row_labels=[f"pass {p['pass']}" for p in res["passes"]],
               title="Bellman-Ford: distances after each pass (negative edges allowed)",
               cell_w=0.72, cell_h=0.46)
    ax.text(0.0, -len(rows) * 0.46 - 0.35,
            "V−1 passes relax every edge; a further improvement would prove a negative cycle.",
            fontsize=8.6, color=WARM)
    register("DIAG_DAA_U3_BELLMAN_01", 3, "bellman_ford", "Bellman-Ford pass by pass distances",
             ["bellman-ford", "bellman ford", "negative weight", "passes", "negative cycle"],
             "Distance estimates after each Bellman-Ford pass on a graph with negative edges.", fig)


def huffman_diagram():
    fig, ax = new_fig(7.0, 3.8)
    nodes = {"100": (0, 1.6), "45:a": (-1.4, 0.7), "55": (1.4, 0.7),
             "25": (0.5, -0.2), "30": (2.4, -0.2),
             "12:c": (0.0, -1.1), "13:b": (1.1, -1.1),
             "14:d": (1.9, -1.1), "16:e": (3.0, -1.1)}
    edges = [("100", "45:a"), ("100", "55"), ("55", "25"), ("55", "30"),
             ("25", "12:c"), ("25", "13:b"), ("30", "14:d"), ("30", "16:e")]
    for u, v in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.1)
        ax.text((x1 + x2) / 2 - 0.12, (y1 + y2) / 2, "0" if x2 < x1 else "1",
                fontsize=8.5, color=ACCENT, weight="bold")
    for name, (x, y) in nodes.items():
        leaf = ":" in name
        ax.add_patch(Circle((x, y), 0.25, facecolor="#e8f6ec" if leaf else "white",
                            edgecolor=INK, lw=1.3, zorder=3))
        ax.text(x, y, name.replace(":", "\n"), ha="center", va="center",
                fontsize=7.2, zorder=4)
    ax.text(0.9, -1.9, "codes: a=0, c=100, b=101, d=110, e=111", fontsize=9,
            color=WARM, weight="bold", ha="center")
    ax.set_title("Huffman tree: merge the two lowest frequencies repeatedly",
                 fontsize=10.5, weight="bold")
    ax.set_xlim(-2.2, 3.8)
    ax.set_ylim(-2.3, 2.1)
    register("DIAG_DAA_U3_HUFFMAN_01", 3, "huffman_coding", "Huffman coding tree with codewords",
             ["huffman", "prefix code", "codeword", "frequency", "encoding tree"],
             "Huffman tree built by merging lowest frequencies, with the resulting prefix codes.",
             fig)


# --------------------------------------------------------------------------
# Unit 4
# --------------------------------------------------------------------------

def knapsack_table_diagram():
    res = knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7)
    fig, ax = new_fig(7.6, 3.4)
    draw_table(ax, res["table"], col_labels=list(range(8)),
               row_labels=["i=0"] + [f"i={i}" for i in range(1, 5)],
               title=f"0/1 knapsack table, optimal value {res['optimal_value']} "
                     f"(items {res['chosen_items']})",
               highlight={(4, 7)}, cell_w=0.62, cell_h=0.46)
    ax.text(0.0, -5 * 0.46 - 0.3,
            "w = [1,3,4,5], v = [1,4,5,7], W = 7    "
            "V[i][w] = max(V[i-1][w], v_i + V[i-1][w-w_i])",
            fontsize=8.5, color=MUTED)
    register("DIAG_DAA_U4_KNAPSACK_TABLE_01", 4, "knapsack_01", "0/1 knapsack dynamic programming table",
             ["0/1 knapsack", "knapsack table", "dp table", "knapsack matrix"],
             "Completed 0/1 knapsack DP table with the optimal cell highlighted.", fig)


def lcs_table_diagram():
    res = lcs("ABCBDAB", "BDCABA")
    sym = res["arrow_symbols"]
    fig, ax = new_fig(6.4, 3.8)
    arrows = {}
    for i in range(1, len(res["table"])):
        for j in range(1, len(res["table"][0])):
            arrows[(i, j)] = sym[res["arrows"][i][j]]
    path = {(s["i"], s["j"]) for s in res["traceback"]}
    draw_table(ax, res["table"], col_labels=["-"] + list("BDCABA"),
               row_labels=["-"] + list("ABCBDAB"),
               title=f"LCS table for ABCBDAB and BDCABA, length {res['length']} "
                     f"(LCS = {res['lcs']})",
               highlight=path, arrows=arrows, cell_w=0.72, cell_h=0.46)
    register("DIAG_DAA_U4_LCS_TABLE_01", 4, "lcs", "LCS length table with direction arrows",
             ["lcs", "longest common subsequence", "direction arrow", "lcs table", "traceback"],
             "LCS table with direction arrows; the highlighted cells are the traceback path.", fig)


def floyd_matrix_diagram():
    W = [[0, 3, None, 7], [8, 0, 2, None], [5, None, 0, 1], [2, None, None, 0]]
    res = floyd_warshall(W)
    show = [0, 1, 2, 4]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))
    for ax, k in zip(axes, show):
        ax.set_axis_off()
        st = res["stages"][k]
        draw_table(ax, st["matrix"], col_labels=[1, 2, 3, 4], row_labels=[1, 2, 3, 4],
                   title=f"D^({st['k']})", cell_w=0.56, cell_h=0.44)
    fig.suptitle("Floyd-Warshall: d[i][j]^(k) = min(d[i][j]^(k-1), d[i][k]^(k-1) + d[k][j]^(k-1))",
                 fontsize=10.5, weight="bold")
    register("DIAG_DAA_U4_FLOYD_MATRIX_01", 4, "floyd_warshall", "Floyd-Warshall distance matrices",
             ["floyd", "warshall", "distance matrix", "all pair shortest path", "d matrix"],
             "Distance matrices D(0), D(1), D(2) and D(4) of the Floyd-Warshall algorithm.", fig)


def mcm_table_diagram():
    res = matrix_chain([30, 35, 15, 5, 10, 20, 25])
    m = [[("" if j < i else ("0" if i == j else res["m_table"][i][j]))
          for j in range(1, 7)] for i in range(1, 7)]
    fig, ax = new_fig(7.0, 3.6)
    draw_table(ax, m, col_labels=[f"j={j}" for j in range(1, 7)],
               row_labels=[f"i={i}" for i in range(1, 7)],
               title=f"Matrix chain m table, minimum cost {res['min_cost']}",
               highlight={(0, 5)}, cell_w=0.78, cell_h=0.44)
    ax.text(0.0, -6 * 0.44 - 0.32,
            f"p = {res['dims']}   optimal parenthesisation {res['parenthesization']}",
            fontsize=8.6, color=WARM)
    register("DIAG_DAA_U4_MCM_TABLE_01", 4, "matrix_chain", "matrix chain multiplication cost table",
             ["matrix chain", "m table", "parenthesization", "parenthesisation", "scalar multiplication"],
             "Matrix chain m table with the minimum multiplication cost and parenthesisation.", fig)


def nqueen_tree_diagram():
    fig, ax = new_fig(7.6, 3.6)
    nodes = {"root": (0, 1.6), "1": (-2.2, 0.6), "2": (-0.7, 0.6), "3": (0.7, 0.6), "4": (2.2, 0.6),
             "1-3": (-2.6, -0.4), "1-4": (-1.6, -0.4), "2-4": (-0.7, -0.4),
             "1-3-x": (-2.6, -1.4), "1-4-2": (-1.6, -1.4), "2-4-1": (-0.7, -1.4),
             "2-4-1-3": (-0.7, -2.4)}
    edges = [("root", "1"), ("root", "2"), ("root", "3"), ("root", "4"),
             ("1", "1-3"), ("1", "1-4"), ("2", "2-4"),
             ("1-3", "1-3-x"), ("1-4", "1-4-2"), ("2-4", "2-4-1"), ("2-4-1", "2-4-1-3")]
    dead = {"1-3-x", "1-4-2"}
    for u, v in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.0)
    for name, (x, y) in nodes.items():
        colour = "#fdeaea" if name in dead else ("#e8f6ec" if name == "2-4-1-3" else "white")
        ax.add_patch(Circle((x, y), 0.3, facecolor=colour, edgecolor=INK, lw=1.2, zorder=3))
        ax.text(x, y, "x" if name in dead else ("root" if name == "root" else name.split("-")[-1]),
                ha="center", va="center", fontsize=8, zorder=4)
    ax.text(2.4, -1.4, "x = pruned by the\nbounding function", fontsize=8.5, color=WARM)
    ax.text(-0.7, -3.0, "solution (2,4,1,3)", fontsize=9, color="#1e7a46",
            weight="bold", ha="center")
    ax.set_title("4-Queens state space tree explored by backtracking",
                 fontsize=10.5, weight="bold")
    ax.set_xlim(-3.4, 4.2)
    ax.set_ylim(-3.4, 2.1)
    register("DIAG_DAA_U4_NQUEEN_TREE_01", 4, "backtracking", "4-queens state space tree",
             ["n-queen", "n queens", "state space tree", "backtracking", "pruning", "bounding function"],
             "State space tree for 4-Queens showing pruned branches and one solution.", fig)


# --------------------------------------------------------------------------
# Unit 5
# --------------------------------------------------------------------------

def kmp_prefix_diagram():
    pattern = "ABABAC"
    pi = [0, 0, 1, 2, 3, 0]
    fig, ax = new_fig(6.6, 2.4)
    draw_table(ax, [list(pattern), pi], col_labels=list(range(1, 7)),
               row_labels=["pattern", "π[i]"],
               title="KMP prefix function for the pattern ABABAC",
               cell_w=0.72, cell_h=0.5)
    ax.text(0.0, -2 * 0.5 - 0.3,
            "π[i] = length of the longest proper prefix of P[1..i] that is also a suffix",
            fontsize=8.5, color=MUTED)
    register("DIAG_DAA_U5_KMP_PREFIX_01", 5, "string_matching", "KMP prefix function table",
             ["kmp", "prefix function", "failure function", "knuth-morris-pratt"],
             "Prefix function table used by KMP to shift the pattern without re-comparing.", fig)


def np_classes_diagram():
    fig, ax = new_fig(6.4, 3.6)
    ax.add_patch(plt.Circle((0.0, 0.0), 1.75, facecolor="#eaf3fb", edgecolor=INK, lw=1.3))
    ax.add_patch(plt.Circle((-0.55, 0.0), 0.85, facecolor="#e8f6ec", edgecolor=INK, lw=1.3))
    ax.add_patch(plt.Circle((0.9, 0.0), 0.75, facecolor="#fff4e0", edgecolor=INK, lw=1.3))
    ax.add_patch(plt.Circle((1.9, 0.0), 1.3, facecolor="none", edgecolor=WARM, lw=1.6, ls="--"))
    ax.text(-0.55, 0.0, "P", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(0.9, 0.0, "NPC", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.0, 1.45, "NP", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(2.6, 1.1, "NP-hard", ha="center", va="center", fontsize=10,
            weight="bold", color=WARM)
    ax.text(0.0, -2.25, "Assuming P ≠ NP. NP-complete = NP ∩ NP-hard.",
            ha="center", fontsize=9, color=MUTED)
    ax.set_title("Relationship between P, NP, NP-complete and NP-hard",
                 fontsize=10.5, weight="bold")
    ax.set_xlim(-2.4, 3.6)
    ax.set_ylim(-2.6, 2.0)
    ax.set_aspect("equal")
    register("DIAG_DAA_U5_NP_CLASSES_01", 5, "np_completeness", "P NP NP-complete NP-hard relationship",
             ["np-complete", "np-hard", "class p", "class np", "complexity classes"],
             "Euler diagram of P, NP, NP-complete and NP-hard under the assumption P != NP.", fig)


def main() -> None:
    print("generating diagrams")
    for fn in (
        recursion_tree_diagram, master_cases_diagram, merge_sort_diagram,
        avl_ll_diagram, avl_lr_diagram, rb_insert_diagram, btree_split_diagram,
        binomial_heap_diagram,
        bfs_dfs_diagram, kruskal_diagram, prim_diagram, dijkstra_diagram,
        bellman_ford_diagram, huffman_diagram,
        knapsack_table_diagram, lcs_table_diagram, floyd_matrix_diagram,
        mcm_table_diagram, nqueen_tree_diagram,
        kmp_prefix_diagram, np_classes_diagram,
    ):
        fn()

    ids = [d["id"] for d in REGISTRY]
    assert len(ids) == len(set(ids)), "duplicate diagram id"
    (OUT / "diagrams.json").write_text(
        json.dumps({"id_format": "DIAG_DAA_U{unit}_{CONCEPT}_{NN}",
                    "count": len(REGISTRY), "diagrams": REGISTRY}, indent=2) + "\n"
    )
    print(f"wrote {len(REGISTRY)} diagrams and registry to {OUT}")


if __name__ == "__main__":
    main()
