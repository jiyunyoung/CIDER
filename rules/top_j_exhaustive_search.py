"""
Beam Search Baseline for LDPC Demixing

Algorithm:
    beam = { empty_path }
    for i in range(N):
        proposals = AMP_topL(i)   # [(symbol, logp), ...]
        new_beam = []
        for path in beam:
            for assignment in cartesian(proposals, repeat=K):
                p = path.extend(assignment)
                p.loglik += sum(lp for (_, lp) in assignment)
                if parity_impossible(p, i):
                    continue
                new_beam.append(p)
        beam = top_B(new_beam)
    valid = [p for p in beam if p.all_checks_zero()]
    return argmax(valid, key=lambda p: p.loglik)

Complexity: O(N * B * L^K) where B=beam width, L=proposal width, K=num codewords
"""

import torch
import torch.nn.functional as F
import numpy as np
from itertools import permutations, product
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import heapq

from utils.gf import get_gf


@dataclass
class Path:
    """A partial hypothesis for K codewords up to position i."""
    symbols: List[List[int]]  # [K, i] symbols assigned so far
    loglik: float             # cumulative log-likelihood
    partial_syndromes: np.ndarray  # [K, M] running syndrome sums

    def __lt__(self, other):
        # For heap: higher loglik = better
        return self.loglik > other.loglik


class BeamSearchDecoder:
    """
    Beam search decoder for LDPC demixing.

    Args:
        Q: Field size (GF(Q))
        N: Codeword length
        K: Number of codewords to demix
        M: Number of parity checks
        H: Parity check matrix [M, N]
        beam_width: Number of paths to keep (B)
        proposal_width: Number of symbol proposals per position (L)
    """

    def __init__(
        self,
        Q: int,
        N: int,
        K: int,
        M: int,
        H: np.ndarray,
        beam_width: int = 100,
        proposal_width: int = 8,
    ):
        self.Q = Q
        self.N = N
        self.K = K
        self.M = M
        self.H = H  # [M, N]
        self.beam_width = beam_width
        self.proposal_width = proposal_width

        # GF(Q) arithmetic
        self.gf = get_gf(Q)

        # Precompute which checks involve which positions
        # check_positions[m] = list of positions involved in check m
        self.check_positions = []
        for m in range(M):
            self.check_positions.append(np.where(H[m] != 0)[0].tolist())

        # position_checks[n] = list of checks involving position n
        self.position_checks = []
        for n in range(N):
            self.position_checks.append(np.where(H[:, n] != 0)[0].tolist())

    def gf_mul(self, a: int, b: int) -> int:
        """GF(Q) multiplication."""
        if a == 0 or b == 0:
            return 0
        return self.gf.exp_table[self.gf.log_table[a] + self.gf.log_table[b]]

    def gf_add(self, a: int, b: int) -> int:
        """GF(Q) addition (XOR for binary extension fields)."""
        return a ^ b

    def get_proposals(self, Y: np.ndarray, position: int, top_l: int) -> List[Tuple[int, float]]:
        """
        Get top-L symbol proposals for a position based on observation Y.

        Simple version: use Y[position] as soft scores directly.
        Could be enhanced with AMP/BP message passing.

        Args:
            Y: [N, Q] soft scores (can be log-probs or raw scores)
            position: which position
            top_l: number of proposals

        Returns:
            List of (symbol, log_probability) tuples
        """
        scores = Y[position]  # [Q]

        # Check if scores are log-probs (mostly negative) or raw probs (positive)
        if scores.max() <= 0:
            # Already log-probabilities, use directly
            log_probs = scores - np.logaddexp.reduce(scores)  # normalize
        else:
            # Raw probabilities, convert to log
            log_probs = np.log(scores + 1e-10) - np.log(scores.sum() + 1e-10)

        # Get top-L indices
        top_indices = np.argsort(log_probs)[-top_l:][::-1]

        return [(int(idx), float(log_probs[idx])) for idx in top_indices]

    def parity_impossible(self, path: Path, position: int) -> bool:
        """
        Check if any parity constraint is already impossible to satisfy.

        A check is impossible if:
        - All its positions have been assigned
        - But the syndrome is non-zero

        Args:
            path: current partial path
            position: just-assigned position (0-indexed)

        Returns:
            True if some parity check is already violated
        """
        # Check all parity constraints involving this position
        for m in self.position_checks[position]:
            check_pos = self.check_positions[m]

            # Are all positions in this check assigned?
            max_assigned = position + 1  # positions 0..position are assigned
            all_assigned = all(p < max_assigned for p in check_pos)

            if all_assigned:
                # Check if syndrome is zero for all K codewords
                for k in range(self.K):
                    if path.partial_syndromes[k, m] != 0:
                        return True

        return False

    def extend_path(
        self,
        path: Path,
        position: int,
        assignment: Tuple[Tuple[int, float], ...],
    ) -> Path:
        """
        Extend a path with a new assignment at the given position.

        Args:
            path: current path
            position: position to assign
            assignment: tuple of (symbol, logp) for each of K codewords

        Returns:
            New extended path
        """
        # Copy symbols
        new_symbols = [s.copy() for s in path.symbols]
        for k, (sym, _) in enumerate(assignment):
            new_symbols[k].append(sym)

        # Update log-likelihood (count unique symbols only to avoid diagonal bias)
        unique_lps = {sym: lp for sym, lp in assignment}
        new_loglik = path.loglik + sum(unique_lps.values())

        # Update partial syndromes
        new_syndromes = path.partial_syndromes.copy()
        for k, (sym, _) in enumerate(assignment):
            for m in self.position_checks[position]:
                coef = self.H[m, position]
                contrib = self.gf_mul(coef, sym)
                new_syndromes[k, m] = self.gf_add(new_syndromes[k, m], contrib)

        return Path(
            symbols=new_symbols,
            loglik=new_loglik,
            partial_syndromes=new_syndromes,
        )

    def all_checks_zero(self, path: Path) -> bool:
        """Check if all parity checks are satisfied for all codewords."""
        return np.all(path.partial_syndromes == 0)

    def decode(self, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Decode K codewords from observation Y using exhaustive search.

        Exhaustive approach:
        1. Get top-L candidates at each position
        2. Generate all L^N possible codewords
        3. Filter by parity check
        4. Pick best pair that covers the observations

        Args:
            Y: [N, Q] soft scores

        Returns:
            codewords: [K, N] decoded codewords
            loglik: log-likelihood of best solution
        """
        L = self.proposal_width

        # Get top-L candidates at each position
        top_L_idx = np.argsort(Y, axis=1)[:, -L:]  # [N, L]
        top_L_set = [set(top_L_idx[i]) for i in range(self.N)]
        top1 = top_L_idx[:, -1]  # [N] - best symbol at each position

        # Generate all L^N codewords and filter by parity
        valid_codewords = []
        valid_logliks = []

        for bits in product(range(L), repeat=self.N):
            # Form codeword: bits[i] selects which of top-L to use at position i
            cw = np.array([top_L_idx[i, -(bits[i]+1)] for i in range(self.N)])

            # Check parity
            syn = self._gf_syndrome(cw)
            if np.all(syn == 0):
                # Compute log-likelihood
                loglik = sum(Y[i, cw[i]] for i in range(self.N))
                valid_codewords.append(cw)
                valid_logliks.append(loglik)

        # Handle cases based on number of valid codewords
        if len(valid_codewords) == 0:
            # No valid codeword - return top-1 duplicated
            cw = top1
            return np.stack([cw, cw], axis=0), 0.0

        elif len(valid_codewords) == 1:
            # Only 1 valid codeword - duplicate it
            cw = valid_codewords[0]
            return np.stack([cw, cw], axis=0), valid_logliks[0]

        else:
            # Find best pair that covers top-L at each position
            from itertools import combinations

            best_pair = None
            best_score = -1
            best_loglik = float('-inf')

            for i, j in combinations(range(len(valid_codewords)), 2):
                cw1, cw2 = valid_codewords[i], valid_codewords[j]

                # Score: positions where pair covers top-L or both are top-1 (collision)
                coverage = 0
                for pos in range(self.N):
                    pair_set = {cw1[pos], cw2[pos]}
                    if pair_set == top_L_set[pos]:
                        coverage += 1  # Covers both top candidates
                    elif cw1[pos] == cw2[pos] == top1[pos]:
                        coverage += 1  # Collision case: both are top-1

                pair_loglik = valid_logliks[i] + valid_logliks[j]

                # Prefer higher coverage, then higher loglik
                if coverage > best_score or (coverage == best_score and pair_loglik > best_loglik):
                    best_score = coverage
                    best_loglik = pair_loglik
                    best_pair = (cw1, cw2)

            if best_pair is None:
                best_pair = (valid_codewords[0], valid_codewords[1])
                best_loglik = valid_logliks[0] + valid_logliks[1]

            return np.stack(best_pair, axis=0), best_loglik

    def _gf_syndrome(self, codeword: np.ndarray) -> np.ndarray:
        """Compute syndrome H @ codeword in GF(Q)."""
        syn = np.zeros(self.M, dtype=np.int64)
        for m in range(self.M):
            for n in range(self.N):
                if self.H[m, n] != 0 and codeword[n] != 0:
                    # GF multiply
                    prod = self.gf.exp_table[self.gf.log_table[self.H[m, n]] + self.gf.log_table[codeword[n]]]
                    syn[m] ^= prod  # GF add
        return syn

    def decode_batch(self, Y_batch: np.ndarray) -> np.ndarray:
        """
        Decode a batch of observations.

        Args:
            Y_batch: [B, N, Q] soft scores

        Returns:
            codewords: [B, K, N] decoded codewords
        """
        B = Y_batch.shape[0]
        results = []

        for b in range(B):
            codewords, _ = self.decode(Y_batch[b])
            results.append(codewords)

        return np.stack(results, axis=0)


class BeamSearchDemixer:
    """
    PyTorch Lightning-compatible wrapper for beam search decoder.
    """

    def __init__(self, config):
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))
        self.M = config.data.M

        # Beam search parameters
        self.beam_width = config.model.get('beam_width', 100)
        self.proposal_width = config.model.get('proposal_width', 2)

        self.decoder = None  # Initialized when H is set
        self.H_matrix = None

    def set_H_matrix(self, H):
        """Set the parity check matrix and initialize decoder."""
        if isinstance(H, torch.Tensor):
            H = H.cpu().numpy()
        self.H_matrix = H
        self.decoder = BeamSearchDecoder(
            Q=self.Q,
            N=self.N,
            K=self.K,
            M=self.M,
            H=H,
            beam_width=self.beam_width,
            proposal_width=self.proposal_width,
        )

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Decode codewords from observation.

        Args:
            Y: [B, N, Q] soft scores

        Returns:
            codewords: [B, K, N] decoded codewords (as indices)
        """
        assert self.decoder is not None, "Must call set_H_matrix first"

        Y_np = Y.cpu().numpy()
        codewords = self.decoder.decode_batch(Y_np)

        return torch.from_numpy(codewords).to(Y.device)

    def __call__(self, Y: torch.Tensor) -> torch.Tensor:
        return self.forward(Y)
