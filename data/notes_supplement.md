# Supplementary DAA Notes (authored for syllabus coverage)

These notes cover KCS-503 syllabus topics that are absent from the supplied
`1038678511-Daa-Aktu-Notes.txt`. They are authored material, tagged in the index
with `source = notes_supplement.md` so every answer citing them is traceable.

## 4.2b Longest Common Subsequence

A subsequence of a string is obtained by deleting zero or more characters without
changing the order of the remaining characters. The Longest Common Subsequence
(LCS) problem takes two sequences X = x1 x2 ... xm and Y = y1 y2 ... yn and asks
for a longest sequence that is a subsequence of both.

LCS has optimal substructure. Let c[i][j] be the length of an LCS of the prefixes
X_i and Y_j. Then:

    c[i][j] = 0                              if i = 0 or j = 0
    c[i][j] = c[i-1][j-1] + 1                if i,j > 0 and x_i = y_j
    c[i][j] = max(c[i-1][j], c[i][j-1])      if i,j > 0 and x_i != y_j

The algorithm fills an (m+1) x (n+1) table row by row. A parallel table b[i][j]
stores a direction arrow used to recover the subsequence:

    diagonal arrow  when x_i = y_j        (the character belongs to the LCS)
    up arrow        when c[i-1][j] >= c[i][j-1]
    left arrow      otherwise

Reading the arrows backwards from b[m][n] and emitting a character on every
diagonal arrow produces one LCS in reverse order.

Time complexity is Theta(m n) because each of the m n table entries is computed
in constant time. Space is Theta(m n), reducible to Theta(min(m, n)) if only the
LCS length is required.

Worked example: X = ABCBDAB, Y = BDCABA. The completed table gives c[7][6] = 4
and the arrows recover the LCS BCBA (BDAB and BCAB are also longest common
subsequences of length 4).

## 4.4b All Pairs Shortest Paths: Floyd-Warshall

Floyd-Warshall computes shortest path distances between every pair of vertices of
a weighted directed graph. Negative edge weights are allowed; negative weight
cycles are not.

Let d[i][j]^(k) be the weight of a shortest path from i to j whose intermediate
vertices all lie in {1, 2, ..., k}. The recurrence is:

    d[i][j]^(0) = w(i, j)
    d[i][j]^(k) = min( d[i][j]^(k-1), d[i][k]^(k-1) + d[k][j]^(k-1) )

The algorithm is three nested loops over k, i and j, so the running time is
Theta(V^3) and the space is Theta(V^2). The matrix D^(n) holds the final all
pairs shortest path distances. A diagonal entry that becomes negative signals a
negative weight cycle.

    FLOYD-WARSHALL(W)
    1. n = rows[W]; D = W
    2. for k = 1 to n
    3.     for i = 1 to n
    4.         for j = 1 to n
    5.             D[i][j] = min(D[i][j], D[i][k] + D[k][j])
    6. return D

Warshall's algorithm is the same triple loop with min/plus replaced by OR/AND,
and it computes the transitive closure of a directed graph in Theta(V^3).

Compared with running Bellman-Ford from every source, which costs O(V^2 E),
Floyd-Warshall is preferable on dense graphs.

## 4.6 Backtracking

Backtracking performs a depth-first traversal of an implicit state space tree. A
partial solution is extended one component at a time. A bounding function tests
whether the partial solution can still lead to a feasible complete solution; when
it cannot, the whole subtree is pruned and the search backtracks.

### n-Queens

Place n queens on an n x n board so that no two share a row, column or diagonal.
Queens are placed one per row. Queen k in column j conflicts with an earlier queen
i in column x[i] when x[i] = j (same column) or |x[i] - j| = |i - k| (same
diagonal). For n = 4 the solutions are (2, 4, 1, 3) and (3, 1, 4, 2). The state
space tree has at most n! leaves; pruning keeps the explored part far smaller.

### Graph Colouring

Assign one of m colours to every vertex so that adjacent vertices differ. Colours
are tried in order for each vertex and the bounding function rejects a colour that
already appears on an adjacent vertex. The chromatic number is the smallest m for
which a colouring exists. Worst case running time is O(m^V V).

### Sum of Subsets

Given positive weights w1 ... wn and a target M, find subsets summing to M. With
the weights sorted, the node at level i is pruned when the running sum plus w_i
exceeds M, or when the running sum plus the remaining weights is below M.

### Hamiltonian Cycles

Find a cycle visiting every vertex exactly once. The next vertex must be adjacent
to the current one and not already on the path, and the last vertex must be
adjacent to the start.

## 4.7 Branch and Bound

Branch and bound also searches a state space tree, but it is used for optimisation
rather than feasibility, and it explores nodes in an order driven by a cost bound.
Each live node carries a lower bound on the cost of any solution in its subtree. A
node whose bound is no better than the best complete solution found so far is
killed. Search strategies include FIFO (breadth first), LIFO (depth first) and
least cost, in which the live node with the smallest bound becomes the next
E-node.

### Travelling Salesman Problem by branch and bound

The lower bound comes from matrix reduction. Reduce the cost matrix by
subtracting the row minimum from each row and then the column minimum from each
column; the total subtracted is the bound at the root. Branching on edge (i, j)
sets row i and column j to infinity, sets entry (j, i) to infinity, and reduces
the matrix again; the child's bound is the parent bound plus the edge cost plus
the new reduction. TSP remains NP-hard, so branch and bound prunes the search but
does not change the worst case.

## 2.4 Tries and Skip Lists

A trie (retrieval tree) stores strings by path. Each edge carries a character, so
a node's position spells a prefix and all descendants of a node share that prefix.
Search, insert and delete take O(L) time for a string of length L, independent of
how many strings the trie holds. Tries are used for dictionaries, prefix search
and autocomplete. A compressed trie contracts every chain of single-child nodes
into one edge.

A skip list is a randomised alternative to a balanced tree. It is a stack of
sorted linked lists: level 0 holds every element, and an element in level i is
promoted to level i+1 with probability p, usually 1/2. Search starts at the top
level, moves right while the next key is smaller than the target, and drops down a
level otherwise. The expected number of levels is O(log n) and the expected cost
of search, insert and delete is O(log n), while the worst case is O(n). Skip lists
need no rotations, which makes them simpler to implement than red-black trees.

## 2.5 AVL Trees and Rotations

An AVL tree is a binary search tree in which the balance factor of every node, the
height of its left subtree minus the height of its right subtree, lies in
{-1, 0, +1}. An insertion or deletion that violates this is repaired by one or two
rotations at the lowest unbalanced node:

    LL case  (left child, left subtree)   single right rotation
    RR case  (right child, right subtree) single left rotation
    LR case  (left child, right subtree)  left rotation then right rotation
    RL case  (right child, left subtree)  right rotation then left rotation

Worked example of the LL case. Insert 30, then 20, then 10 into an empty AVL
tree. After inserting 30 the tree is a single node with balance factor 0. After
inserting 20 the root 30 has balance factor +1, which is still allowed. After
inserting 10 the left subtree of 30 has height 2 and its right subtree has height
0, so the balance factor of 30 becomes +2 and the AVL property is violated at 30.
The offending node 10 is in the left subtree of the left child 20, so this is the
LL case and one single right rotation about 30 repairs it. Node 20 becomes the
root with 10 as its left child and 30 as its right child. After the rotation the
balance factors are 0 for 20, 0 for 10 and 0 for 30, and the height drops from 3
to 2.

Worked example of the LR case. Insert 30, then 10, then 20. The balance factor of
30 becomes +2 while the new node 20 sits in the right subtree of the left child
10, which is the LR case. A left rotation at 10 turns the configuration into the
LL case, and a right rotation at 30 then finishes the repair, leaving 20 as the
root with children 10 and 30 and every balance factor 0.

Deletion uses the same four cases, but where an insertion needs at most one
rotation, a deletion may require rotations at every node on the path back to the
root, so a deletion costs O(log n) rotations in the worst case.

A rotation runs in O(1) time and the height of an AVL tree with n nodes is
O(log n), so search, insert and delete are all O(log n). Red-black trees use the
same rotation primitives but a weaker balance condition, allowing fewer rotations
per update at the cost of greater height.

## 1.4 Sorting in Linear Time

Any comparison sort needs Omega(n log n) comparisons in the worst case, because a
decision tree with n! leaves has height at least log2(n!) = Omega(n log n). The
following sorts beat that bound by not comparing elements to each other.

Counting sort assumes keys in the range 0 to k. It counts the occurrences of each
key, converts the counts to running totals, then places each element directly. It
runs in Theta(n + k) and is stable.

Radix sort sorts d-digit numbers by applying a stable sort to each digit position,
least significant digit first, in Theta(d(n + k)) time.

Bucket sort distributes n inputs drawn uniformly from [0, 1) into n buckets, sorts
each bucket, then concatenates. The expected running time is Theta(n).

## 3.1b Divide and Conquer: Strassen and Convex Hull

Strassen's algorithm multiplies two n x n matrices using 7 recursive
multiplications of n/2 x n/2 submatrices instead of 8, plus Theta(n^2) additions.
Its recurrence T(n) = 7 T(n/2) + Theta(n^2) solves to Theta(n^log2(7)), about
Theta(n^2.81), beating the Theta(n^3) of the ordinary algorithm.

The convex hull of a point set is the smallest convex polygon containing every
point. The divide and conquer method splits the points by x coordinate, computes
the hull of each half recursively, and merges them by finding the upper and lower
common tangents in O(n) time, giving T(n) = 2 T(n/2) + O(n) = O(n log n).

## 5.4 Randomized Algorithms

A randomized algorithm makes random choices during execution. A Las Vegas
algorithm is always correct and its running time is a random variable; randomized
quicksort, which picks a random pivot, is a Las Vegas algorithm with expected
running time O(n log n). A Monte Carlo algorithm always finishes within a bound
but may be wrong with small probability; repeating it lowers the error
probability. Randomization removes the dependence on any particular bad input,
because no fixed input can force the bad case.

## 5.5 Algebraic Computation and the Fast Fourier Transform

Multiplying two polynomials of degree n by the direct method costs Theta(n^2). The
Fast Fourier Transform evaluates a polynomial at the n complex nth roots of unity
in Theta(n log n) time by divide and conquer, splitting the coefficients into even
and odd indexed halves. Polynomial multiplication then becomes: transform both
polynomials to point value form with the FFT, multiply the values pointwise in
Theta(n) time, and interpolate back with an inverse FFT. Total cost is
Theta(n log n).

## 3.3b Job Sequencing with Deadlines

Each job i has a deadline d_i and a profit p_i, earned only if the job finishes
by its deadline. Every job takes one unit of time and one job runs at a time.
The goal is to choose a subset and an order maximising total profit.

The greedy method sorts the jobs by profit in non-increasing order, then places
each job in the latest free slot at or before its deadline. Scheduling a job as
late as possible keeps the earlier slots free for jobs with tighter deadlines. If
no free slot at or before the deadline exists, the job is rejected.

    JOB-SEQUENCING(jobs, n)
    1. sort jobs by profit in non-increasing order
    2. slot[1..max_deadline] = free
    3. for each job j in sorted order
    4.     for t = min(max_deadline, d_j) down to 1
    5.         if slot[t] is free: assign j to t; break

With an array scan the running time is O(n^2); a disjoint set union over free
slots reduces it to O(n log n). The greedy choice is optimal because exchanging
a scheduled lower-profit job for a rejected higher-profit one never decreases the
total profit.

Worked example: jobs (J1, d=2, p=100), (J2, d=1, p=19), (J3, d=2, p=27),
(J4, d=1, p=25), (J5, d=3, p=15). Sorted by profit: J1, J3, J4, J2, J5. J1 takes
slot 2, J3 takes slot 1, J4 and J2 are rejected, J5 takes slot 3. The schedule is
J3, J1, J5 with total profit 142.
