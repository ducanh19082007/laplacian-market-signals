"""
    Turn ExchangeRateGraph.adjacency (a dict-of-dicts) into dense numpy matrices
    so we can do spectral work: adjacency W, degree D, Laplacian L = D - W.

    ---------------------------------------------------------------------------
    STEP 0 -- FREEZE AN ORDERING
    ---------------------------------------------------------------------------
    A dict has no "row 0 / row 1". A matrix does. So before anything we pin every
    node to an integer index by sorting graph.nodes() (sorted() is deterministic,
    so the same graph always gives the same matrix -- important for eigenvalues).

        self.nodes = [                       index
            ("btc", "Binance"),   # ------->   0
            ("btc", "Kraken"),    # ------->   1
            ("eth", "Binance"),   # ------->   2
            ("eth", "Kraken"),    # ------->   3
            ("xrp", "Binance"),   # ------->   4
        ]
        self.index = {("btc","Binance"): 0, ("btc","Kraken"): 1, ...}

    ---------------------------------------------------------------------------
    STEP 1 -- ADJACENCY MATRIX  W   (this is the "simple matrix" you asked for)
    ---------------------------------------------------------------------------
    W is n x n. Row = source node, Column = destination node.

        W[i][j] = weight of the edge  nodes[i] -> nodes[j]
                = 0.0  if there is NO such edge

    Reading the dict straight across:

        adjacency[("eth","Binance")][("btc","Binance")] = {"weight": w1, ...}
                    row  = index[("eth","Binance")] = 2
                    col  = index[("btc","Binance")] = 0
        =>  W[2][0] = w1

    So the dict from build_from_snapshot() lands as (using the docstring example,
    'x' = some real weight, '0' = missing edge / the None entries you mentioned):

                          TO:  btc@Bin  btc@Krk  eth@Bin  eth@Krk  xrp@Bin
                                (0)      (1)      (2)      (3)      (4)
        FROM btc@Binance (0)  [  0        0        x        0        x   ]
        FROM btc@Kraken  (1)  [  0        0        0        x        0   ]
        FROM eth@Binance (2)  [  x        0        0        x        x   ]
        FROM eth@Kraken  (3)  [  0        x        x        0        0   ]
        FROM xrp@Binance (4)  [  x        0        x        0        0   ]

      * NOT symmetric: eth@Bin -> btc@Bin (a convert leg) exists, and so does the
        reverse, but with a DIFFERENT weight, so W[2][0] != W[0][2].
      * The '0' at, say, [eth@Krk][eth@Bin]... actually that transfer DOES exist;
        the genuinely-missing one from your note (eth@Kraken has no xrp edge) is
        W[3][4] = 0. Any pair with no edge in the dict stays 0 in the matrix.

    ---------------------------------------------------------------------------
    STEP 2 -- DEGREE MATRIX  D
    ---------------------------------------------------------------------------
    Diagonal only: D[i][i] = weighted out-degree of node i = sum of row i of W.
    Everything off-diagonal is 0.

        D[i][i] = sum_j W[i][j]

    ---------------------------------------------------------------------------
    STEP 3 -- LAPLACIAN  L = D - W
    ---------------------------------------------------------------------------
    Subtract elementwise. Diagonal keeps the row-sum, off-diagonal is -W[i][j].
    Every row of L sums to 0 (that's the defining property of a Laplacian, and
    why lambda = 0 is always an eigenvalue). Spectral analysis (lambda_2, the
    negative-real-part check for negative cycles, etc.) all runs on THIS matrix.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..L1_DataProcessing.DataProcessing import ExchangeRateGraph, Node
except ImportError:
    from L1_DataProcessing.DataProcessing import ExchangeRateGraph, Node

#firstly, should initilaizethe STANDARD LAPLACIAN matrix L = D - W, hence, be aware that
#this will do a lot with np now :)
class Laplacian():

    def __init__(
        self,
        graph: "ExchangeRateGraph",
        attr: str = "weight",
        symmetrize: str = "unweighted",
    ) -> None:
        # attr -- which DIRECTED edge scalar fills the raw matrix W. W is kept only
        #   for display / arbitrage reference; it is signed and asymmetric, so it is
        #   NOT what the Laplacian is built from.
        #     "weight" -> -log(rate) value (call graph.log_transform() first)
        #     "rate"   -> raw multiplier
        #
        # symmetrize -- how the raw DIRECTED matrix W is turned into the SYMMETRIC
        #   matrix A the Laplacian is built from. A Laplacian only has real, ordered
        #   eigenvalues (hence a valid Fiedler value / spectral gap) when it is
        #   symmetric, so this step is mandatory. Three modes, kept from the six
        #   surveyed -- each is the best in its lane; citations in affinity_matrix():
        #     "unweighted" -> union (OR) symmetrization, binary presence.
        #                     >>> BEST DEFAULT for the health / OU pipeline. <<<
        #     "average"    -> additive symmetrization (W + W^T)/2 on |weight|.
        #                     Use with a non-negative magnitude (liquidity, 1/spread)
        #                     when you want a smooth lambda_2(t) to feed OU.
        #     "signed"     -> Kunegis signed Laplacian: keeps the +/- of -log(rate),
        #                     so structure reflects arbitrage direction.
        self.graph = graph
        self.attr = attr
        self.symmetrize = symmetrize

        # STEP 0: freeze the node ordering (dict keys -> integer indices).
        self.nodes: List[Node] = graph.nodes()
        self.index: Dict[Node, int] = {node: i for i, node in enumerate(self.nodes)}
        self.n = len(self.nodes)

        # Built lazily by the methods below.
        self.W: Optional[np.ndarray] = None   # raw directed matrix (signed/asymmetric)
        self.A: Optional[np.ndarray] = None   # symmetric non-negative affinity
        self.D: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None
        self.NormalizedL: Optional[np.ndarray] = None
        self.FiedlerVal: float = 0.0
        self.FiedlerVector: Optional[np.ndarray] = None
        self.MarketStrain: float = 0.0

    def adjacency_matrix(self) -> np.ndarray:
        """Reminder of what the adjacentcy matrix looks like:"
                Build the graph from MultiBrokerOrderBook.snapshot():
            snapshot[pair][broker] = {"bid": ..., "ask": ...}

        For each broker that quotes a pair we add the two conversion directions,
        then we stitch venues together with same-asset transfer edges.
        
        Output on self.adjacent examples:
        self.adjacency = {
            ("eth", "Binance"): {
                ("btc", "Binance"): {"rate": 0.0610,        "weight": None, "kind": "convert", "pair": "ethbtc", "broker": "Binance"},
                ("xrp", "Binance"): {"rate": 5089.058...,   "weight": None, "kind": "convert", "pair": "xrpeth",  "broker": "Binance"},
                ("eth", "Kraken"):  {"rate": 1.0,           "weight": None, "kind": "transfer"},
            },
            ("btc", "Binance"): {
                ("eth", "Binance"): {"rate": 16.3666...,    "weight": None, "kind": "convert", "pair": "ethbtc", "broker": "Binance"},
                ("xrp", "Binance"): {"rate": 83263.94...,   "weight": None, "kind": "convert", "pair": "xrpbtc", "broker": "Binance"},
                ("btc", "Kraken"):  {"rate": 1.0,           "weight": None, "kind": "transfer"},
            },
            ("eth", "Kraken"): {
                ("btc", "Kraken"):  {"rate": 0.0600,        "weight": None, "kind": "convert", "pair": "ethbtc", "broker": "Kraken"},
                ("eth", "Binance"): {"rate": 1.0,           "weight": None, "kind": "transfer"},
            },
        }
        """
        W = np.zeros((self.n, self.n), dtype=float)
        for u, nbrs in self.graph.adjacency.items():
            i = self.index[u]
            for v, edge in nbrs.items():
                j = self.index[v]
                value = edge[self.attr]
                # weight is None until log_transform() runs; skip so it stays 0.
                if value is not None:
                    W[i][j] = value
        self.W = W
        return W

    def affinity_matrix(self) -> np.ndarray: #this can be called as similarity/weight matrix
        """STEP 1b: turn the directed W into the SYMMETRIC matrix A the Laplacian uses.

        A Laplacian only has real, ordered eigenvalues -- hence a valid Fiedler value
        lambda_2 -- when it is symmetric. So we must symmetrize the directed graph
        first. This is a standard preprocessing step with a whole family of choices;
        below are the three I kept, each the best in its lane, with the paper each
        comes from. `self.symmetrize` selects one.

        Note: presence is read from the adjacency DICT, not from `W != 0`, because a
        transfer edge has rate 1.0 -> weight 0.0 and would be wrongly dropped by a
        nonzero test. Diagonal is forced to 0 (no self-loops).

        MODES
        -----
        "unweighted" : A_ij = 1 if an edge exists in EITHER direction (union/OR
            symmetrization, binary). Always non-negative -> L is PSD, eigenvalues
            real and >= 0. Simplest and most robust; measures pure connectivity.
            >>> BEST DEFAULT for the health / OU pipeline. <<<
            Ref: the binary union is folklore; what gives lambda_2 its meaning is
                 M. Fiedler, "Algebraic connectivity of graphs," Czechoslovak Math.
                 J. 23(2) (1973), 298-305 -- def. of lambda_2 p. 298, Thm. 3.1
                 p. 300. Directed Cheeger bound: F. Chung, "Laplacians and the
                 Cheeger inequality for directed graphs," Annals of Combinatorics 9
                 (2005), 1-19 -- Laplacian def. Sec. 3 (p. 4), Cheeger Thm. 4.3 (p. 8).

        "average" : A = (M + M^T)/2 with M_ij = |weight of edge i->j|. The textbook
            additive symmetrization. Feed it a genuine non-negative magnitude
            (liquidity, 1/spread) and lambda_2(t) varies smoothly -- the input OU
            wants. On |-log(rate)| it works but is coarse.
            Ref: V. Satuluri & S. Parthasarathy, "Symmetrizations for clustering
                 directed graphs," EDBT 2011, 343-354 -- A_sym=(A+A^T)/2 in Sec. 3.1
                 (p. 345). Also U. von Luxburg, "A Tutorial on Spectral Clustering,"
                 Statistics and Computing 17(4) (2007), 395-416 -- symmetrizing a
                 directed W, Sec. 8 (p. 411).

        "signed" : A = (W + W^T)/2 keeping the sign of -log(rate); the degree then
            uses |A| (done in degree_matrix), giving the SIGNED Laplacian L = D - A.
            Keeps arbitrage direction in the structure instead of discarding it.
            Ref: J. Kunegis et al., "Spectral Analysis of Signed Graphs for
                 Clustering, Prediction and Visualization," SIAM SDM 2010, 559-570 --
                 signed Laplacian L=D-A with D_ii=sum_j|A_ij|, Sec. 3, Eqs. (3)-(4)
                 (p. 561).

        (Page/equation numbers are best-effort locators for the journal versions --
         confirm against your own copy, they shift between preprint and final.)
        """
        W = self.adjacency_matrix() if self.W is None else self.W
        mode = self.symmetrize

        if mode == "unweighted":
            A = np.zeros((self.n, self.n), dtype=float)
            for u, nbrs in self.graph.adjacency.items():
                i = self.index[u]
                for v in nbrs:
                    j = self.index[v]
                    A[i][j] = 1.0
                    A[j][i] = 1.0        # union: either direction links the pair, basically the binary adjacency matrix
        elif mode == "average":
            M = np.abs(W)
            A = 0.5 * (M + M.T)
        elif mode == "signed":
            A = 0.5 * (W + W.T)          # sign preserved; |.| applied in degree_matrix
        else:
            raise ValueError(f"unknown symmetrize mode: {mode!r}")

        np.fill_diagonal(A, 0.0)
        self.A = A
        return A

    def degree_matrix(self) -> np.ndarray:
        """STEP 2: diagonal degree matrix D_ii = sum_j |A_ij|.

        Using |A| (not A) is what makes "signed" mode Kunegis's signed Laplacian; for
        the non-negative modes ("unweighted"/"average") |A| == A, so this is the
        ordinary weighted degree. Either way D_ii >= 0, so D^-1/2 stays real.
        """
        if self.A is None:
            self.affinity_matrix()
        self.D = np.diag(np.abs(self.A).sum(axis=1))
        return self.D

    def laplacian(self) -> np.ndarray:
        """STEP 3: the standard Laplacian L = D - A.

        Built on the symmetric non-negative affinity A (NOT the signed directed W),
        so L is symmetric positive-semidefinite: every eigenvalue is real and >= 0,
        lambda_1 = 0, and lambda_2 (the Fiedler value) is a genuine measure of how
        well-connected the market graph is. Feeding the signed -log(rate) matrix here
        instead would give negative degrees, complex eigenvalues, and no valid lambda_2.
        """
        if self.D is None:
            self.degree_matrix()
        self.L = self.D - self.A
        return self.L

    def NormalizedLaplacian(self) -> np.ndarray:
        """Symmetric normalized Laplacian  L_sym = D^-1/2 L D^-1/2.

        Same D^-1/2 on BOTH sides (D^-1/2 L D^+1/2 is just a similarity transform of
        L with identical eigenvalues -- it does nothing). Its eigenvalues live in
        [0, 2], which makes lambda_2 comparable across graphs of different size/scale.
        Zero-degree (isolated) nodes get 1/sqrt(d) = 0 instead of blowing up inv().

        Ref: F. Chung, "Spectral Graph Theory," CBMS Regional Conf. Series in Math.
             92, AMS (1997) -- L_sym def. Sec. 1.2 (Eq. 1.2, p. 2), eigenvalues in
             [0, 2] Lemma 1.7 (p. 6).
        """
        if self.L is None:
            self.laplacian()
        d = np.diag(self.D).astype(float)
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt)
        self.NormalizedL = D_inv_sqrt @ self.L @ D_inv_sqrt
        return self.NormalizedL 

    def FiedlerValue(self) -> float:
        """The Fiedler value: second-smallest eigenvalue of the normalized Laplacian.

        Uses eigh (the symmetric-matrix solver): it returns REAL eigenvalues already
        sorted ascending, so index 1 is lambda_2 and column 1 is its eigenvector.
        """
        if self.NormalizedL is None:
            self.NormalizedLaplacian()
        eigenvalues, eigenvectors = np.linalg.eigh(self.NormalizedL)
        self.FiedlerVal = float(eigenvalues[1])
        self.FiedlerVector = eigenvectors[:, 1]
        # High strain = market close to fragmenting (lambda_2 near 0).
        self.MarketStrain = 1.0 / self.FiedlerVal if self.FiedlerVal > 1e-12 else np.inf
        return self.FiedlerVal
        



if __name__ == "__main__":
    # Smoke test against L1's own rigged snapshot.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from L1_DataProcessing.DataProcessing import ExchangeRateGraph

    assets = ["btc", "eth", "xrp", "sol"]
    snapshot = {
        "ethbtc": {
            "Binance": {"bid": "0.0610", "ask": "0.0611"},
            "Kraken":  {"bid": "0.0600", "ask": "0.0601"},
        },
        "xrpbtc": {"Binance": {"bid": "0.00001230", "ask": "0.00001201"}},
        "xrpeth": {"Binance": {"bid": "0.00019600", "ask": "0.00019650"}},
    }

    graph = ExchangeRateGraph(assets, transfer_cost=0.0).build_from_snapshot(snapshot).log_transform()

    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    lap = Laplacian(graph, attr="weight", symmetrize="unweighted")
    print("Node ordering (row/col index):")
    for i, node in enumerate(lap.nodes):
        print(f"  {i}: {ExchangeRateGraph.fmt(node)}")

    print("\nW = raw directed matrix (signed -log(rate); reference only, NOT used by L):")
    print(lap.adjacency_matrix())

    # Compare the three symmetrizations on the same graph.
    for mode in ("unweighted", "average", "signed"):
        lap = Laplacian(graph, attr="weight", symmetrize=mode)
        print(f"\n================ symmetrize = {mode!r} ================")
        print("A = symmetric matrix the Laplacian is built from:")
        print(lap.affinity_matrix())
        print("L_norm = D^-1/2 L D^-1/2:")
        print(lap.NormalizedLaplacian())
        print(f"  lambda_2 (Fiedler) = {lap.FiedlerValue():.6f}")
        print(f"  MarketStrain=1/l2  = {lap.MarketStrain:.6f}")