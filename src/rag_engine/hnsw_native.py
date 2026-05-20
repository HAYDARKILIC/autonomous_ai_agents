"""A from-scratch HNSW (Hierarchical Navigable Small World) index.

This implementation is used in the Week 2 notebook to expose the
mathematics that production vector stores hide. Production code in
:mod:`pipeline` uses Chroma, but the algorithms here are faithful to
Malkov & Yashunin (2018, IEEE TPAMI).

Algorithmic summary
-------------------
HNSW maintains an *L*-layer hierarchy of graphs over the vector set.
A new point :math:`x` is assigned a level

.. math::

    \\ell = \\lfloor -\\ln(u) \\cdot m_L \\rfloor, \\quad u \\sim U(0, 1)

where :math:`m_L = 1 / \\ln(M)`. This produces an exponentially decaying
expected layer occupancy: roughly :math:`M^{-\\ell}` of points appear
at layer :math:`\\ell`. The top layer behaves like an entry-point
small-world graph; the bottom layer is a dense kNN graph.

Search is a greedy descent: enter at the top layer, find the locally
nearest neighbor by greedy expansion, then descend to the next layer
and refine. At the bottom layer the search returns the *k* nearest
neighbors with an ``ef``-sized priority queue.

The expected query cost is :math:`O(\\log N)` for uniform data — a
dramatic improvement over the :math:`O(N)` cost of brute-force search.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    vector: np.ndarray
    payload: dict
    neighbors: list[list[int]] = field(default_factory=list)  # per-layer neighbor lists


class HNSWIndex:
    """A pedagogical HNSW implementation over cosine-similarity vectors.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    M : int
        Maximum neighbors per node per layer (above layer 0). Layer 0 uses
        ``M * 2`` per the original paper.
    ef_construction : int
        Dynamic candidate list size during insertion.
    ef_search : int
        Dynamic candidate list size during query.
    seed : int
        RNG seed for reproducibility of level assignment.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 1234,
    ) -> None:
        self.dim = dim
        self.M = M
        self.M0 = M * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = 1.0 / math.log(M)  # level-assignment normalization constant
        self._rng = random.Random(seed)
        self._nodes: list[_Node] = []
        self._entry_point: int | None = None
        self._max_layer: int = -1

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def add(self, vector: np.ndarray, payload: dict) -> int:
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {vector.shape}.")
        level = self._sample_level()
        node_id = len(self._nodes)
        node = _Node(vector=vector, payload=payload,
                     neighbors=[[] for _ in range(level + 1)])
        self._nodes.append(node)

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_layer = level
            return node_id

        # Greedy descent from the top layer down to level+1.
        curr = self._entry_point
        for layer in range(self._max_layer, level, -1):
            curr = self._greedy_search(vector, curr, layer)

        # Connect to ef_construction nearest at every layer up to `level`.
        for layer in range(min(level, self._max_layer), -1, -1):
            candidates = self._search_layer(vector, curr, self.ef_construction, layer)
            m = self.M0 if layer == 0 else self.M
            selected = self._select_neighbors_heuristic(vector, candidates, m)
            node.neighbors[layer] = [c for _, c in selected]
            # Add reciprocal edges; prune over-full neighbor lists.
            for _, neighbor_id in selected:
                self._nodes[neighbor_id].neighbors[layer].append(node_id)
                if len(self._nodes[neighbor_id].neighbors[layer]) > m:
                    self._prune(neighbor_id, layer, m)
            curr = selected[0][1]

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = node_id
        return node_id

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[float, dict]]:
        if self._entry_point is None:
            return []
        curr = self._entry_point
        for layer in range(self._max_layer, 0, -1):
            curr = self._greedy_search(query, curr, layer)
        results = self._search_layer(query, curr, max(self.ef_search, k), layer=0)
        results.sort(key=lambda x: x[0])  # ascending distance
        return [(d, self._nodes[nid].payload) for d, nid in results[:k]]

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _sample_level(self) -> int:
        u = self._rng.random()
        # ℓ = ⌊-ln(u) · m_L⌋
        return int(math.floor(-math.log(u + 1e-12) * self.m_L))

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        # Cosine distance assuming L2-normalized inputs.
        return float(1.0 - np.dot(a, b))

    def _greedy_search(self, query: np.ndarray, entry: int, layer: int) -> int:
        curr = entry
        curr_d = self._distance(query, self._nodes[curr].vector)
        improved = True
        while improved:
            improved = False
            for neighbor in self._nodes[curr].neighbors[layer]:
                d = self._distance(query, self._nodes[neighbor].vector)
                if d < curr_d:
                    curr_d, curr, improved = d, neighbor, True
        return curr

    def _search_layer(
        self, query: np.ndarray, entry: int, ef: int, layer: int
    ) -> list[tuple[float, int]]:
        # Two heaps: candidates (min-heap on dist), results (max-heap on dist).
        entry_d = self._distance(query, self._nodes[entry].vector)
        candidates: list[tuple[float, int]] = [(entry_d, entry)]
        results: list[tuple[float, int]] = [(-entry_d, entry)]
        visited: set[int] = {entry}

        while candidates:
            d, curr = heapq.heappop(candidates)
            worst_in_results = -results[0][0]
            if d > worst_in_results:
                break
            for neighbor in self._nodes[curr].neighbors[layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nd = self._distance(query, self._nodes[neighbor].vector)
                worst_in_results = -results[0][0]
                if nd < worst_in_results or len(results) < ef:
                    heapq.heappush(candidates, (nd, neighbor))
                    heapq.heappush(results, (-nd, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [(-neg_d, nid) for neg_d, nid in results]

    def _select_neighbors_heuristic(
        self, query: np.ndarray, candidates: list[tuple[float, int]], m: int
    ) -> list[tuple[float, int]]:
        """Heuristic selection (Malkov & Yashunin, Alg. 4) preserving diversity."""
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        selected: list[tuple[float, int]] = []
        for d, cand_id in sorted_cands:
            if len(selected) >= m:
                break
            # Accept the candidate only if it is closer to the query than to any
            # already-selected neighbor — this preserves angular diversity.
            cand_vec = self._nodes[cand_id].vector
            ok = True
            for _, s_id in selected:
                if self._distance(cand_vec, self._nodes[s_id].vector) < d:
                    ok = False
                    break
            if ok:
                selected.append((d, cand_id))
        return selected

    def _prune(self, node_id: int, layer: int, m: int) -> None:
        node = self._nodes[node_id]
        scored = [
            (self._distance(node.vector, self._nodes[nid].vector), nid)
            for nid in node.neighbors[layer]
        ]
        kept = self._select_neighbors_heuristic(node.vector, scored, m)
        node.neighbors[layer] = [nid for _, nid in kept]
