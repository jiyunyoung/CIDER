"""
Stitching Decoder for GF(q) Codes (SPARC-style).

Adapted from the tree code stitching decoder in SPARC-based URA
(Fengler et al. 2021, Ebert et al. 2022).

Algorithm:
  1. At each position n, extract top-J candidates from Y[n]
  2. Grow single-codeword paths position-by-position:
     - For each existing path, try each candidate at the next position
     - Update partial GF(q) syndromes
     - Prune if any fully-determined check has nonzero syndrome
  3. Collect all valid codewords (all syndromes = 0)
  4. Rank by log-likelihood, select top-K as the decoded set

This produces K *distinct* codewords naturally — each path grows
from a different starting candidate, so duplicates only occur if
the code structure allows it.

Works with any parity-check matrix H over GF(Q) (LDPC, tree code, RS).
Most effective when H has causal/cascading structure (tree codes)
so that parity checks complete early and pruning is aggressive.

Complexity: O(N * |paths| * J) per sample, where |paths| is bounded
by beam_width. For tree codes with good pruning, |paths| stays small.
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

from utils.gf import get_gf


class StitchingDecoder:
    """
    SPARC-style stitching decoder for GF(q) codes.

    Finds individual valid codewords via beam search with
    incremental parity pruning, then selects the best K.
    """

    def __init__(
        self,
        Q: int,
        N: int,
        K: int,
        M: int,
        H: np.ndarray,
        beam_width: int = 10000,
        proposal_width: int = 3,
    ):
        self.Q = Q
        self.N = N
        self.K = K
        self.M = M
        self.H = np.asarray(H, dtype=int)
        self.beam_width = beam_width
        self.proposal_width = proposal_width

        self.gf = get_gf(Q)

        # Precompute GF multiplication table
        self._mul_table = np.zeros((Q, Q), dtype=int)
        for a in range(Q):
            for b in range(Q):
                if a == 0 or b == 0:
                    self._mul_table[a, b] = 0
                else:
                    self._mul_table[a, b] = self.gf.exp_table[
                        self.gf.log_table[a] + self.gf.log_table[b]
                    ]

        # For each check m, which positions and coefficients?
        self.check_positions = []
        self.check_coeffs_map = []
        for m in range(M):
            positions = []
            coef_map = {}
            for n in range(N):
                if H[m, n] != 0:
                    positions.append(n)
                    coef_map[n] = int(H[m, n])
            self.check_positions.append(positions)
            self.check_coeffs_map.append(coef_map)

        # For each position, which checks?
        self.position_checks = []
        for n in range(N):
            checks = [m for m in range(M) if H[m, n] != 0]
            self.position_checks.append(checks)

        # Optimal position ordering for early pruning
        self.pos_order = self._compute_position_order()

    def _compute_position_order(self) -> List[int]:
        """Greedy ordering that completes parity checks as early as possible."""
        assigned = set()
        order = []
        remaining_checks = set(range(self.M))

        for _ in range(self.N):
            best_pos = -1
            best_score = -1

            for n in range(self.N):
                if n in assigned:
                    continue
                score = 0
                test_assigned = assigned | {n}
                for m in self.position_checks[n]:
                    if m in remaining_checks:
                        if all(p in test_assigned for p in self.check_positions[m]):
                            score += 1
                if score > best_score or (score == best_score and (best_pos == -1 or n < best_pos)):
                    best_score = score
                    best_pos = n

            order.append(best_pos)
            assigned.add(best_pos)

            for m in list(remaining_checks):
                if all(p in assigned for p in self.check_positions[m]):
                    remaining_checks.discard(m)

        return order

    def _get_top_j(self, Y: np.ndarray, position: int) -> List[Tuple[int, float]]:
        """Get top-J symbol candidates at a position."""
        scores = Y[position]
        if scores.max() <= 0:
            log_probs = scores - np.logaddexp.reduce(scores)
        else:
            log_probs = np.log(scores + 1e-10) - np.log(scores.sum() + 1e-10)

        top_indices = np.argsort(log_probs)[-self.proposal_width:][::-1]
        return [(int(idx), float(log_probs[idx])) for idx in top_indices]

    def _find_valid_codewords(self, Y: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Find all valid codewords via beam search with parity stitching.

        Each path represents a SINGLE codeword being built.

        Returns:
            List of (codeword [N], loglik) for all parity-valid codewords found.
        """
        N, M = self.N, self.M

        # Each beam entry: (loglik, symbol_assignments [(pos, sym), ...], syndromes [M])
        init_syn = np.zeros(M, dtype=int)
        beam = [(0.0, [], init_syn)]

        assigned_so_far = set()

        for step, pos in enumerate(self.pos_order):
            assigned_so_far.add(pos)
            proposals = self._get_top_j(Y, pos)

            new_beam = []
            for loglik, assignments, syndromes in beam:
                for sym, lp in proposals:
                    new_loglik = loglik + lp
                    new_syn = syndromes.copy()

                    # Update syndromes
                    for m in self.position_checks[pos]:
                        coef = self.check_coeffs_map[m][pos]
                        contrib = self._mul_table[coef, sym]
                        new_syn[m] ^= contrib

                    # Prune if any fully-determined check is violated
                    pruned = False
                    for m in self.position_checks[pos]:
                        if all(p in assigned_so_far for p in self.check_positions[m]):
                            if new_syn[m] != 0:
                                pruned = True
                                break

                    if pruned:
                        continue

                    new_assignments = assignments + [(pos, sym)]
                    new_beam.append((new_loglik, new_assignments, new_syn))

            # Keep top beam_width paths
            new_beam.sort(key=lambda x: -x[0])
            beam = new_beam[:self.beam_width]

            if len(beam) == 0:
                break

        # Collect valid codewords (all syndromes = 0)
        valid_codewords = []
        for loglik, assignments, syndromes in beam:
            if np.all(syndromes == 0) and len(assignments) == N:
                cw = np.zeros(N, dtype=int)
                for pos, sym in assignments:
                    cw[pos] = sym
                valid_codewords.append((cw, loglik))

        # Sort by loglik descending
        valid_codewords.sort(key=lambda x: -x[1])

        return valid_codewords

    def _select_top_k(
        self,
        valid_codewords: List[Tuple[np.ndarray, float]],
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Select K distinct codewords from the valid set.

        Greedy: pick top-1 by loglik, then pick the next codeword
        that is most different from already-selected ones.

        Returns:
            codewords: [K, N]
        """
        K, N = self.K, self.N

        if len(valid_codewords) == 0:
            # Fallback: argmax of Y duplicated
            top1 = np.argmax(Y, axis=-1)
            return np.stack([top1] * K, axis=0)

        if len(valid_codewords) <= K:
            # Not enough valid codewords — pad with duplicates
            selected = [cw for cw, _ in valid_codewords]
            while len(selected) < K:
                selected.append(selected[-1].copy())
            return np.stack(selected[:K], axis=0)

        # Greedy diverse selection
        selected = [valid_codewords[0][0]]  # best by loglik

        for _ in range(1, K):
            best_idx = -1
            best_score = -float('inf')

            for i, (cw, ll) in enumerate(valid_codewords):
                # Skip if already selected
                if any(np.array_equal(cw, s) for s in selected):
                    continue
                # Score: loglik + diversity bonus (min hamming to selected)
                min_hamming = min(np.sum(cw != s) for s in selected)
                score = ll + 0.5 * min_hamming  # balance likelihood and diversity
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                selected.append(valid_codewords[best_idx][0])
            else:
                # No more distinct codewords
                selected.append(selected[-1].copy())

        return np.stack(selected, axis=0)

    def decode(self, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Decode K codewords from observation Y.

        1. Find all valid single codewords via parity-pruned beam search
        2. Select top-K distinct codewords

        Args:
            Y: [N, Q] soft scores

        Returns:
            codewords: [K, N] decoded codewords
            loglik: log-likelihood of selected set
        """
        valid_codewords = self._find_valid_codewords(Y)
        codewords = self._select_top_k(valid_codewords, Y)
        total_ll = sum(
            valid_codewords[i][1] if i < len(valid_codewords) else 0.0
            for i in range(self.K)
        )
        return codewords, total_ll

    def decode_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of observations."""
        B = Y_batch.shape[0]
        results = []
        for b in range(B):
            cw, _ = self.decode(Y_batch[b])
            results.append(cw)
        return np.stack(results, axis=0)
