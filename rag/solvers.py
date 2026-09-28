"""Exact numeric solvers for DAA problems.

Every solver returns plain JSON-serialisable data including the intermediate
tables, so the language model narrates steps it did not compute itself. The
arithmetic is done here; the model never does it.
"""

from __future__ import annotations

import math
from fractions import Fraction

INF = float("inf")


# --------------------------------------------------------------------------
# Recurrences
# --------------------------------------------------------------------------

def _pow_str(exp: Fraction) -> str:
    if exp == 0:
        return "1"
    if exp == 1:
        return "n"
    if exp.denominator == 1:
        return f"n^{exp.numerator}"
    return f"n^({exp.numerator}/{exp.denominator})"


def _log_str(power: int) -> str:
    if power == 0:
        return ""
    if power == 1:
        return " log n"
    return f" log^{power} n"


def master_theorem(a: int, b: int, f_exponent: float = 1.0, log_power: int = 0) -> dict:
    """Solve T(n) = a*T(n/b) + f(n) with f(n) = n^f_exponent * log^log_power(n).

    Args:
        a: Number of subproblems, must be >= 1.
        b: Factor the input is divided by, must be > 1.
        f_exponent: Exponent k in the driving function f(n) = n^k * log^p(n).
        log_power: Power p of the logarithm in the driving function.

    Returns:
        Case number, the comparison performed, and the asymptotic bound.
    """
    if a < 1:
        raise ValueError("a must be >= 1")
    if b <= 1:
        raise ValueError("b must be > 1")

    k = Fraction(str(f_exponent))
    crit = math.log(a, b)

    # Exact comparison: a vs b^k avoids floating point error on log_b(a).
    lhs = Fraction(a)
    rhs = Fraction(b) ** k if k.denominator == 1 else Fraction(str(b ** float(k)))

    steps = [
        f"Recurrence: T(n) = {a} T(n/{b}) + f(n), with f(n) = {_pow_str(k)}{_log_str(log_power)}",
        f"Critical exponent: log_b(a) = log_{b}({a}) = {crit:.4f}",
        f"Compare f(n) = {_pow_str(k)}{_log_str(log_power)} with n^log_b(a) = n^{crit:.4f}",
    ]

    if lhs > rhs:
        case = 1
        bound = f"Theta(n^log_{b}({a}))"
        if abs(crit - round(crit)) < 1e-9:
            bound = f"Theta({_pow_str(Fraction(round(crit)))})"
        steps.append(
            f"Since {float(k):.4f} < {crit:.4f}, f(n) grows polynomially slower: Case 1 applies."
        )
        steps.append(f"Therefore T(n) = {bound}.")
        comparison = f"f(n) = O(n^(log_b(a) - eps)), i.e. {float(k):.4f} < {crit:.4f}"
    elif lhs == rhs:
        case = 2
        bound = f"Theta({_pow_str(k)}{_log_str(log_power + 1)})"
        steps.append(
            f"Since {float(k):.4f} = {crit:.4f}, f(n) matches n^log_b(a): Case 2 applies."
        )
        steps.append(
            f"Case 2 multiplies by one extra logarithmic factor, giving T(n) = {bound}."
        )
        comparison = f"f(n) = Theta(n^log_b(a) log^{log_power} n), i.e. {float(k):.4f} = {crit:.4f}"
    else:
        case = 3
        bound = f"Theta({_pow_str(k)}{_log_str(log_power)})"
        steps.append(
            f"Since {float(k):.4f} > {crit:.4f}, f(n) grows polynomially faster: Case 3 applies."
        )
        steps.append(
            f"Regularity condition a f(n/b) <= c f(n) holds for c = {a}/{b}^{float(k):.4f} < 1."
        )
        steps.append(f"Therefore T(n) = {bound}.")
        comparison = f"f(n) = Omega(n^(log_b(a) + eps)), i.e. {float(k):.4f} > {crit:.4f}"

    return {
        "recurrence": f"T(n) = {a} T(n/{b}) + {_pow_str(k)}{_log_str(log_power)}",
        "a": a,
        "b": b,
        "f": f"{_pow_str(k)}{_log_str(log_power)}",
        "log_b_a": round(crit, 4),
        "case": case,
        "comparison": comparison,
        "bound": bound,
        "steps": steps,
    }


def recursion_tree(a: int, b: int, f_exponent: float = 1.0, levels: int = 4) -> dict:
    """Expand T(n) = a*T(n/b) + n^f_exponent as a recursion tree.

    Args:
        a: Branching factor of the tree.
        b: Factor the subproblem size shrinks by at each level.
        f_exponent: Exponent k in the per-node cost n^k.
        levels: Number of levels to expand explicitly.

    Returns:
        Per-level node count, subproblem size, cost, tree height and total cost.
    """
    k = Fraction(str(f_exponent))
    rows = []
    for i in range(levels):
        nodes = f"{a}^{i}" if i else "1"
        size = f"n/{b}^{i}" if i else "n"
        cost_coeff = Fraction(a) ** i / (Fraction(b) ** (Fraction(i) * k))
        rows.append(
            {
                "level": i,
                "nodes": nodes,
                "subproblem_size": size,
                "cost_per_node": f"({size})^{float(k):g}",
                "level_cost": f"({a}/{b}^{float(k):g})^{i} * n^{float(k):g}",
                "level_cost_coefficient": f"{float(cost_coeff):.6g}",
            }
        )

    ratio = Fraction(a) / (Fraction(b) ** k) if k.denominator == 1 else None
    crit = math.log(a, b)
    height = f"log_{b}(n)"
    leaves = f"n^log_{b}({a}) = n^{crit:.4f}"

    if ratio is not None and ratio > 1:
        verdict = "Level costs increase geometrically, so the leaves dominate."
        total = f"Theta(n^log_{b}({a}))"
    elif ratio is not None and ratio == 1:
        verdict = "Every level costs the same, so the total is the level cost times the height."
        total = f"Theta({_pow_str(k)} log n)"
    else:
        verdict = "Level costs decrease geometrically, so the root dominates."
        total = f"Theta({_pow_str(k)})"

    return {
        "recurrence": f"T(n) = {a} T(n/{b}) + {_pow_str(k)}",
        "levels": rows,
        "height": height,
        "leaf_count": leaves,
        "verdict": verdict,
        "total": total,
    }


# --------------------------------------------------------------------------
# Dynamic programming
# --------------------------------------------------------------------------

def knapsack_01(weights: list[int], values: list[int], capacity: int) -> dict:
    """Solve the 0/1 knapsack problem by dynamic programming.

    Args:
        weights: Weight of each item, in item order.
        values: Value of each item, in item order.
        capacity: Capacity of the knapsack.

    Returns:
        The full (n+1) x (capacity+1) DP table, optimal value and chosen items.
    """
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    n = len(weights)
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        for w in range(capacity + 1):
            skip = table[i - 1][w]
            take = table[i - 1][w - wi] + vi if wi <= w else None
            table[i][w] = skip if take is None else max(skip, take)

    chosen, w = [], capacity
    for i in range(n, 0, -1):
        if table[i][w] != table[i - 1][w]:
            chosen.append(i)
            w -= weights[i - 1]
    chosen.reverse()

    return {
        "weights": weights,
        "values": values,
        "capacity": capacity,
        "recurrence": "V[i][w] = max(V[i-1][w], v_i + V[i-1][w - w_i]) if w_i <= w else V[i-1][w]",
        "table": table,
        "row_labels": ["i=0 (no items)"] + [f"i={i} (w={weights[i-1]}, v={values[i-1]})" for i in range(1, n + 1)],
        "optimal_value": table[n][capacity],
        "chosen_items": chosen,
        "chosen_weights": [weights[i - 1] for i in chosen],
        "chosen_values": [values[i - 1] for i in chosen],
    }


def fractional_knapsack(weights: list[int], values: list[int], capacity: float) -> dict:
    """Solve the fractional knapsack problem by the greedy method.

    Args:
        weights: Weight of each item, in item order.
        values: Value of each item, in item order.
        capacity: Capacity of the knapsack.

    Returns:
        Value/weight ratios, the greedy order and the optimal fractional value.
    """
    n = len(weights)
    items = [
        {"item": i + 1, "weight": weights[i], "value": values[i],
         "ratio": round(values[i] / weights[i], 4)}
        for i in range(n)
    ]
    order = sorted(items, key=lambda d: -d["ratio"])
    remaining = float(capacity)
    total = 0.0
    steps = []
    for it in order:
        if remaining <= 0:
            steps.append({**it, "fraction": 0.0, "value_taken": 0.0, "remaining": 0.0})
            continue
        frac = 1.0 if it["weight"] <= remaining else remaining / it["weight"]
        taken = it["value"] * frac
        remaining -= it["weight"] * frac
        total += taken
        steps.append({**it, "fraction": round(frac, 4),
                      "value_taken": round(taken, 4), "remaining": round(remaining, 4)})

    return {
        "weights": weights,
        "values": values,
        "capacity": capacity,
        "greedy_order": [it["item"] for it in order],
        "steps": steps,
        "optimal_value": round(total, 4),
    }


def lcs(x: str, y: str) -> dict:
    """Compute the longest common subsequence of two strings.

    Args:
        x: First sequence.
        y: Second sequence.

    Returns:
        The c length table, the b arrow table, the LCS length and one LCS.
    """
    m, n = len(x), len(y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    b = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
                b[i][j] = "diag"
            elif c[i - 1][j] >= c[i][j - 1]:
                c[i][j] = c[i - 1][j]
                b[i][j] = "up"
            else:
                c[i][j] = c[i][j - 1]
                b[i][j] = "left"

    out, i, j, path = [], m, n, []
    while i > 0 and j > 0:
        path.append({"i": i, "j": j, "arrow": b[i][j]})
        if b[i][j] == "diag":
            out.append(x[i - 1])
            i, j = i - 1, j - 1
        elif b[i][j] == "up":
            i -= 1
        else:
            j -= 1
    out.reverse()
    path.reverse()

    return {
        "x": x,
        "y": y,
        "recurrence": "c[i][j] = c[i-1][j-1]+1 if x_i = y_j else max(c[i-1][j], c[i][j-1])",
        "table": c,
        "arrows": b,
        "arrow_symbols": {"diag": "↖", "up": "↑", "left": "←"},
        "length": c[m][n],
        "lcs": "".join(out),
        "traceback": path,
        "complexity": "Theta(mn)",
    }


def matrix_chain(dims: list[int]) -> dict:
    """Find the optimal parenthesisation for a chain of matrix multiplications.

    Args:
        dims: Dimension array p of length n+1, so matrix A_i is p[i-1] x p[i].

    Returns:
        The m cost table, the s split table, the minimum cost and the
        optimal parenthesisation.
    """
    n = len(dims) - 1
    if n < 1:
        raise ValueError("dims must contain at least two entries")
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = INF
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k

    def build(i: int, j: int) -> str:
        if i == j:
            return f"A{i}"
        return f"({build(i, s[i][j])}{build(s[i][j] + 1, j)})"

    return {
        "dims": dims,
        "matrices": [f"A{i}: {dims[i-1]}x{dims[i]}" for i in range(1, n + 1)],
        "recurrence": "m[i][j] = min over k of ( m[i][k] + m[k+1][j] + p_(i-1) p_k p_j )",
        "m_table": [[(None if v == INF else int(v)) for v in row] for row in m],
        "s_table": s,
        "min_cost": int(m[1][n]),
        "parenthesization": build(1, n),
        "complexity": "Theta(n^3)",
    }


def floyd_warshall(matrix: list[list[float]]) -> dict:
    """Compute all pairs shortest paths with the Floyd-Warshall algorithm.

    Args:
        matrix: Square weight matrix, using null or a very large number for
            absent edges and 0 on the diagonal.

    Returns:
        The intermediate distance matrix after each value of k, plus the final
        matrix and any negative cycle detection.
    """
    n = len(matrix)
    d = [[INF if (v is None) else float(v) for v in row] for row in matrix]
    for i in range(n):
        d[i][i] = min(d[i][i], 0.0)

    def snap(mat):
        return [[(None if v == INF else (int(v) if float(v).is_integer() else v))
                 for v in row] for row in mat]

    stages = [{"k": 0, "label": "D^(0) (initial weights)", "matrix": snap(d)}]
    for k in range(n):
        changed = []
        for i in range(n):
            for j in range(n):
                alt = d[i][k] + d[k][j]
                if alt < d[i][j]:
                    changed.append(
                        {"i": i + 1, "j": j + 1,
                         "old": None if d[i][j] == INF else d[i][j],
                         "new": alt,
                         "via": k + 1}
                    )
                    d[i][j] = alt
        stages.append({"k": k + 1, "label": f"D^({k+1}) (intermediate vertices from {{1..{k+1}}})",
                       "matrix": snap(d), "updates": changed})

    negative_cycle = any(d[i][i] < 0 for i in range(n))
    return {
        "input": matrix,
        "n": n,
        "recurrence": "d[i][j]^(k) = min( d[i][j]^(k-1), d[i][k]^(k-1) + d[k][j]^(k-1) )",
        "stages": stages,
        "final": snap(d),
        "negative_cycle": negative_cycle,
        "complexity": "Theta(V^3)",
    }


# --------------------------------------------------------------------------
# Graph algorithms
# --------------------------------------------------------------------------

def dijkstra(graph: dict, source: str) -> dict:
    """Trace Dijkstra's single source shortest path algorithm.

    Args:
        graph: Adjacency mapping, for example {"A": {"B": 4}, "B": {}}.
        source: Label of the source vertex.

    Returns:
        A per-iteration trace of extracted vertices and relaxations, plus the
        final distances and predecessor tree.
    """
    if source not in graph:
        raise ValueError(f"source {source!r} is not a vertex of the graph")
    if any(w < 0 for u in graph for w in graph[u].values()):
        raise ValueError("Dijkstra requires non-negative edge weights; use bellman_ford")

    dist = {v: INF for v in graph}
    prev = {v: None for v in graph}
    dist[source] = 0.0
    visited: list[str] = []
    trace = []

    while len(visited) < len(graph):
        unvisited = {v: dist[v] for v in graph if v not in visited}
        u = min(unvisited, key=unvisited.get)
        if dist[u] == INF:
            break
        visited.append(u)
        relaxations = []
        for v, w in sorted(graph[u].items()):
            if v in visited:
                continue
            alt = dist[u] + w
            if alt < dist[v]:
                relaxations.append(
                    {"edge": f"{u}->{v}", "weight": w,
                     "old": None if dist[v] == INF else dist[v], "new": alt}
                )
                dist[v] = alt
                prev[v] = u
        trace.append(
            {
                "iteration": len(visited),
                "extracted": u,
                "distance_of_extracted": dist[u],
                "relaxations": relaxations,
                "distances": {k: (None if val == INF else val) for k, val in dist.items()},
                "visited": list(visited),
            }
        )

    return {
        "source": source,
        "relaxation_rule": "if d[u] + w(u,v) < d[v] then d[v] = d[u] + w(u,v) and pi[v] = u",
        "trace": trace,
        "distances": {k: (None if v == INF else v) for k, v in dist.items()},
        "predecessors": prev,
        "complexity": "O((V + E) log V) with a binary heap",
    }


def bellman_ford(graph: dict, source: str) -> dict:
    """Trace the Bellman-Ford shortest path algorithm, including negative edges.

    Args:
        graph: Adjacency mapping, for example {"A": {"B": -2}, "B": {}}.
        source: Label of the source vertex.

    Returns:
        Distances after each of the V-1 passes and the negative cycle verdict.
    """
    if source not in graph:
        raise ValueError(f"source {source!r} is not a vertex of the graph")
    vertices = list(graph)
    edges = [(u, v, w) for u in vertices for v, w in graph[u].items()]
    dist = {v: INF for v in vertices}
    dist[source] = 0.0
    passes = []

    for p in range(1, len(vertices)):
        relaxations = []
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                relaxations.append(
                    {"edge": f"{u}->{v}", "weight": w,
                     "old": None if dist[v] == INF else dist[v], "new": dist[u] + w}
                )
                dist[v] = dist[u] + w
        passes.append(
            {"pass": p, "relaxations": relaxations,
             "distances": {k: (None if val == INF else val) for k, val in dist.items()}}
        )
        if not relaxations:
            break

    negative_cycle = any(
        dist[u] != INF and dist[u] + w < dist[v] for u, v, w in edges
    )

    return {
        "source": source,
        "edge_count": len(edges),
        "relaxation_rule": "relax every edge V-1 times; a further improvement proves a negative cycle",
        "passes": passes,
        "distances": {k: (None if v == INF else v) for k, v in dist.items()},
        "negative_cycle": negative_cycle,
        "complexity": "O(V E)",
    }


SOLVERS = {
    "master_theorem": master_theorem,
    "recursion_tree": recursion_tree,
    "knapsack_01": knapsack_01,
    "fractional_knapsack": fractional_knapsack,
    "lcs": lcs,
    "matrix_chain": matrix_chain,
    "floyd_warshall": floyd_warshall,
    "dijkstra": dijkstra,
    "bellman_ford": bellman_ford,
}
