"""
Factorized Iterative BP Decoder for LDPC Demixing

Factorized approach: instead of tracking joint tuples (s1, s2),
track independent beliefs per user with "explain away" mechanism.

Algorithm:
    Initialize: belief[k][n] = Y[n] for all users

    Iterate:
        1. LDPC BP for user 1 → posterior q1
        2. Explain away: belief[2] ∝ Y / q1
        3. LDPC BP for user 2 → posterior q2
        4. Explain away: belief[1] ∝ Y / q2
        5. Repeat until convergence

Key idea: When user 1 claims symbol a with high probability,
user 2's belief for a decreases, forcing different codewords.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from utils.gf import get_gf


@dataclass
class LDPCGraph:
    """LDPC graph structure for Q-ary codes over GF(Q)."""
    H: np.ndarray  # [M, N] parity check matrix
    Q: int

    def __post_init__(self):
        self.M, self.N = self.H.shape
        self.gf = get_gf(self.Q)

        # Build adjacency lists
        self.check_to_vars: List[List[int]] = [[] for _ in range(self.M)]
        self.check_coeffs: List[List[int]] = [[] for _ in range(self.M)]
        self.var_to_checks: List[List[int]] = [[] for _ in range(self.N)]
        self.var_coeffs: List[List[int]] = [[] for _ in range(self.N)]

        for m in range(self.M):
            for n in range(self.N):
                h = int(self.H[m, n])
                if h != 0:
                    self.check_to_vars[m].append(n)
                    self.check_coeffs[m].append(h)
                    self.var_to_checks[n].append(m)
                    self.var_coeffs[n].append(h)


def normalize_log(logp: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Convert log probs to normalized probs."""
    logp = np.asarray(logp, dtype=float)
    m = np.max(logp)
    p = np.exp(logp - m)
    s = p.sum()
    if s < eps:
        return np.ones_like(p) / len(p)
    return p / s


def normalize_prob(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Normalize probability distribution."""
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s < eps:
        return np.ones_like(p) / len(p)
    return p / s


def _walsh_hadamard_transform(f: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard Transform for XOR convolution over GF(2^m).

    Transforms f of length Q=2^m so that pointwise multiplication
    in the transform domain corresponds to XOR convolution.
    WHT is its own inverse (up to scaling by 1/Q).

    Supports batched input: f can be [..., Q] with arbitrary leading dims.
    """
    a = f.copy()
    Q = a.shape[-1]
    h = 1
    while h < Q:
        # Reshape last dim to (..., Q/(2h), 2, h) for vectorized butterfly
        shape = a.shape[:-1] + (Q // (2 * h), 2, h)
        a = a.reshape(shape)
        x = a[..., 0, :].copy()
        y = a[..., 1, :].copy()
        a[..., 0, :] = x + y
        a[..., 1, :] = x - y
        a = a.reshape(f.shape)
        h *= 2
    return a


class FactorizedBPDecoder:
    """
    Factorized BP decoder for K-user LDPC demixing.

    Each user runs independent LDPC BP, with "explain away"
    updates between users to encourage different codewords.
    """

    def __init__(
        self,
        Q: int,
        N: int,
        K: int,
        M: int,
        H: np.ndarray,
        max_iters: int = 50,
        damping: float = 0.1,
        explain_strength: float = 1.0,
        eps: float = 1e-10,
        use_wht: bool = True,
    ):
        """
        Args:
            Q: Field size GF(Q)
            N: Codeword length
            K: Number of users to demix
            M: Number of parity checks
            H: Parity check matrix [M, N]
            max_iters: Maximum BP iterations
            damping: Message damping factor (0-1)
            explain_strength: How strongly to explain away (0-1)
            eps: Numerical stability constant
            use_wht: If True, use Walsh-Hadamard Transform O(Q log Q);
                     if False, use naive convolution O(Q^2)
        """
        self.Q = Q
        self.N = N
        self.K = K
        self.M = M
        self.H = np.asarray(H, dtype=int)
        self.max_iters = max_iters
        self.damping = damping
        self.explain_strength = explain_strength
        self.eps = eps
        self.use_wht = use_wht

        # Build graph structure
        self.graph = LDPCGraph(H=self.H, Q=Q)
        self.gf = self.graph.gf

        # Precompute GF multiplication lookup table: mul_table[a, b] = a * b
        self._mul_table = np.zeros((Q, Q), dtype=int)
        for a in range(Q):
            for b in range(Q):
                if a == 0 or b == 0:
                    self._mul_table[a, b] = 0
                else:
                    self._mul_table[a, b] = self.gf.exp_table[
                        self.gf.log_table[a] + self.gf.log_table[b]
                    ]

        # Precompute GF inverse table: inv_table[a] = a^{-1}
        self._inv_table = np.zeros(Q, dtype=int)
        for a in range(1, Q):
            log_a = self.gf.log_table[a]
            log_inv = (Q - 1 - log_a) % (Q - 1)
            if log_inv == 0 and log_a != 0:
                log_inv = Q - 1
            self._inv_table[a] = self.gf.exp_table[log_inv]

        # Precompute GF permutation tables for each nonzero element:
        # perm_table[h] = array where perm_table[h][x] = h * x
        self._perm_table = np.zeros((Q, Q), dtype=int)
        for h in range(Q):
            for x in range(Q):
                self._perm_table[h, x] = self._mul_table[h, x]

    def _gf_mul(self, a: int, b: int) -> int:
        """GF(Q) multiplication."""
        return self._mul_table[a, b]

    def _gf_inv(self, a: int) -> int:
        """GF(Q) multiplicative inverse."""
        if a == 0:
            raise ValueError("No inverse for 0")
        return self._inv_table[a]

    def _check_to_var_msg(
        self,
        check_idx: int,
        target_var_idx: int,
        var_to_check_msgs: np.ndarray,  # [N, Q]
    ) -> np.ndarray:
        """
        Compute check-to-variable message using Walsh-Hadamard Transform.

        For check m: sum_n H[m,n] * x[n] = 0 in GF(Q)
        XOR convolution via WHT: O(Q log Q) instead of O(Q^2).
        """
        Q = self.Q
        vars_in_check = self.graph.check_to_vars[check_idx]
        coeffs = self.graph.check_coeffs[check_idx]

        # Find position of target in check's variable list
        target_pos = None
        for i, v in enumerate(vars_in_check):
            if v == target_var_idx:
                target_pos = i
                break

        if target_pos is None:
            return np.ones(Q) / Q

        # Collect incoming messages (excluding target)
        other_msgs = []
        other_coeffs = []
        for i, (v, h) in enumerate(zip(vars_in_check, coeffs)):
            if i != target_pos:
                other_msgs.append(var_to_check_msgs[v])
                other_coeffs.append(h)

        if len(other_msgs) == 0:
            return np.ones(Q) / Q

        # Transform to z domain using vectorized permutation: z = h * x
        # p_z[z] = p_x[h^{-1} * z]  =>  p_z = p_x[perm[h_inv]]
        transformed = []
        for msg, h in zip(other_msgs, other_coeffs):
            h_inv = self._inv_table[h]
            perm = self._perm_table[h_inv]  # perm[z] = h_inv * z
            p_z = normalize_prob(msg[perm], self.eps)
            transformed.append(p_z)

        if self.use_wht:
            # XOR convolution via Walsh-Hadamard Transform: O(Q log Q)
            conv_wht = _walsh_hadamard_transform(transformed[0])
            for p_z in transformed[1:]:
                conv_wht *= _walsh_hadamard_transform(p_z)
            conv = _walsh_hadamard_transform(conv_wht) / Q
            conv = normalize_prob(np.maximum(conv, 0), self.eps)
        else:
            # Naive XOR convolution: O(Q^2)
            conv = transformed[0].copy()
            for p_z in transformed[1:]:
                conv_new = np.zeros(Q)
                for z1 in range(Q):
                    for z2 in range(Q):
                        conv_new[z1 ^ z2] += conv[z1] * p_z[z2]
                conv = normalize_prob(conv_new, self.eps)

        # Transform back: x = h_target^{-1} * z  =>  out[x] = conv[h_target * x]
        h_target = coeffs[target_pos]
        perm_fwd = self._perm_table[h_target]  # perm_fwd[x] = h_target * x
        out = conv[perm_fwd]

        return normalize_prob(out, self.eps)

    def _ldpc_bp_iteration(
        self,
        channel_llr: np.ndarray,  # [N, Q] channel evidence (log domain)
        v2c: np.ndarray,  # [N, M, Q] var-to-check messages
        c2v: np.ndarray,  # [M, N, Q] check-to-var messages
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One iteration of LDPC BP.

        Returns:
            v2c_new, c2v_new, posteriors
        """
        Q, N, M = self.Q, self.N, self.M
        damp = self.damping
        eps = self.eps

        # Convert channel LLR to probabilities (vectorized)
        logp = channel_llr.astype(float)
        logp_max = logp.max(axis=-1, keepdims=True)
        channel_prob = np.exp(logp - logp_max)
        row_sums = channel_prob.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums < eps, 1.0, row_sums)
        channel_prob = channel_prob / row_sums

        # Variable to check messages
        v2c_new = np.zeros_like(v2c)
        for n in range(N):
            checks = self.graph.var_to_checks[n]
            # Compute product of all incoming check messages
            prod = channel_prob[n].copy()
            for m in checks:
                prod *= np.maximum(c2v[m, n], eps)
            prod = normalize_prob(prod, eps)

            # Message to each check excludes that check's message
            for m in checks:
                msg = prod / np.maximum(c2v[m, n], eps)
                msg = normalize_prob(msg, eps)
                v2c_new[n, m] = (1 - damp) * v2c[n, m] + damp * msg
                v2c_new[n, m] = normalize_prob(v2c_new[n, m], eps)

        # Check to variable messages
        c2v_new = np.zeros_like(c2v)
        for m in range(M):
            vars_in_check = self.graph.check_to_vars[m]
            for n in vars_in_check:
                msg = self._check_to_var_msg(m, n, v2c_new[:, m, :])
                c2v_new[m, n] = (1 - damp) * c2v[m, n] + damp * msg
                c2v_new[m, n] = normalize_prob(c2v_new[m, n], eps)

        # Compute posteriors
        posteriors = np.zeros((N, Q))
        for n in range(N):
            post = channel_prob[n].copy()
            for m in self.graph.var_to_checks[n]:
                post *= np.maximum(c2v_new[m, n], eps)
            posteriors[n] = normalize_prob(post, eps)

        return v2c_new, c2v_new, posteriors

    def _check_syndrome(self, codeword: np.ndarray) -> bool:
        """Check if codeword satisfies all parity checks."""
        for m in range(self.M):
            syndrome = 0
            for n, h in zip(self.graph.check_to_vars[m], self.graph.check_coeffs[m]):
                syndrome ^= self._gf_mul(h, codeword[n])
            if syndrome != 0:
                return False
        return True

    def _count_syndrome_errors(self, codeword: np.ndarray) -> int:
        """Count number of unsatisfied parity checks."""
        errors = 0
        for m in range(self.M):
            syndrome = 0
            for n, h in zip(self.graph.check_to_vars[m], self.graph.check_coeffs[m]):
                syndrome ^= self._gf_mul(h, codeword[n])
            if syndrome != 0:
                errors += 1
        return errors

    def _decode_multi_ordering(self, Y: np.ndarray, num_orderings: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Try multiple user orderings and pick the one with lowest total syndrome errors.

        For K=3, the ordering significantly affects which user gets degraded channel.
        """
        import itertools

        K = self.K
        best_codewords = None
        best_posteriors = None
        best_score = float('inf')

        # Generate orderings to try
        if num_orderings >= np.math.factorial(K):
            # Try all permutations
            orderings = list(itertools.permutations(range(K)))
        else:
            # Random sample of orderings
            orderings = [list(range(K))]  # Always include default
            all_perms = list(itertools.permutations(range(K)))
            np.random.shuffle(all_perms)
            for perm in all_perms[:num_orderings - 1]:
                if list(perm) not in orderings:
                    orderings.append(list(perm))

        for order in orderings:
            codewords, posteriors = self._decode_with_order(Y, order)

            # Score: total syndrome errors across all users
            total_errors = sum(self._count_syndrome_errors(codewords[k]) for k in range(K))

            # Bonus: prefer unique codewords
            unique_cw = len(set(tuple(codewords[k]) for k in range(K)))
            if unique_cw < K:
                total_errors += 1000  # Penalty for duplicates

            if total_errors < best_score:
                best_score = total_errors
                best_codewords = codewords
                best_posteriors = posteriors

                if total_errors == 0:
                    break  # Perfect decode

        return best_codewords, best_posteriors

    def _decode_with_order(self, Y: np.ndarray, order: list) -> Tuple[np.ndarray, np.ndarray]:
        """Decode with a specific user ordering."""
        Q, N, K, M = self.Q, self.N, self.K, self.M
        eps = self.eps

        # Initialize channel evidence for each user
        channel = np.zeros((K, N, Q))
        for k in range(K):
            channel[k] = Y.copy()

        # Initialize messages for each user
        v2c = np.ones((K, N, M, Q)) / Q
        c2v = np.ones((K, M, N, Q)) / Q

        # Initialize posteriors
        posteriors = np.zeros((K, N, Q))
        for k in range(K):
            for n in range(N):
                posteriors[k, n] = normalize_log(Y[n], eps)

        # Iterative decoding with specified order
        for iter_idx in range(self.max_iters):
            # Process users in specified order
            for k in order:
                # Step 1: Run BP for user k
                v2c[k], c2v[k], posteriors[k] = self._ldpc_bp_iteration(
                    channel[k], v2c[k], c2v[k]
                )

                # Step 2: Update other users' channel evidence (vectorized)
                Y_prob = np.exp(Y - Y.max(axis=-1, keepdims=True))
                Y_prob = Y_prob / (Y_prob.sum(axis=-1, keepdims=True) + eps)

                for k2 in range(K):
                    if k2 != k:
                        new_channel = Y_prob.copy()  # [N, Q]
                        for k3 in order[:order.index(k) + 1]:
                            if k3 != k2:
                                explain = 1.0 - self.explain_strength * posteriors[k3] + self.explain_strength / Q
                                new_channel *= explain
                        row_sums = new_channel.sum(axis=-1, keepdims=True)
                        row_sums = np.where(row_sums < eps, 1.0, row_sums)
                        new_channel = new_channel / row_sums
                        channel[k2] = np.log(new_channel + eps)

            # Check convergence
            hard = np.argmax(posteriors, axis=-1)
            all_valid = all(self._check_syndrome(hard[k]) for k in range(K))
            if all_valid:
                unique_codewords = set(tuple(hard[k]) for k in range(K))
                if len(unique_codewords) == K:
                    break

        codewords = np.argmax(posteriors, axis=-1)
        return codewords, posteriors

    def decode(self, Y: np.ndarray, num_orderings: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode K codewords from observation Y.

        Uses SEQUENTIAL updates: run user k's BP, then immediately
        update other users' channel evidence before their BP runs.
        This breaks the symmetry that causes both users to converge
        to the same codeword.

        Args:
            Y: [N, Q] soft scores (log probabilities)
            num_orderings: Try multiple random orderings and pick best (for K>=3)

        Returns:
            codewords: [K, N] decoded codewords
            posteriors: [K, N, Q] posterior probabilities
        """
        if num_orderings > 1 and self.K >= 3:
            return self._decode_multi_ordering(Y, num_orderings)

        Q, N, K, M = self.Q, self.N, self.K, self.M
        eps = self.eps

        # Initialize channel evidence for each user
        channel = np.zeros((K, N, Q))
        for k in range(K):
            channel[k] = Y.copy()

        # Initialize messages for each user
        v2c = np.ones((K, N, M, Q)) / Q
        c2v = np.ones((K, M, N, Q)) / Q

        # Initialize posteriors
        posteriors = np.zeros((K, N, Q))
        for k in range(K):
            for n in range(N):
                posteriors[k, n] = normalize_log(Y[n], eps)

        # Iterative decoding with SEQUENTIAL updates
        for iter_idx in range(self.max_iters):
            # Process each user sequentially
            for k in range(K):
                # Step 1: Run BP for user k with current channel evidence
                v2c[k], c2v[k], posteriors[k] = self._ldpc_bp_iteration(
                    channel[k], v2c[k], c2v[k]
                )

                # Step 2: IMMEDIATELY update other users' channel evidence
                # This ensures next user sees the explain-away effect
                # Vectorized over N positions
                Y_prob = np.exp(Y - Y.max(axis=-1, keepdims=True))
                Y_prob = Y_prob / (Y_prob.sum(axis=-1, keepdims=True) + eps)

                for k2 in range(K):
                    if k2 != k:
                        new_channel = Y_prob.copy()  # [N, Q]

                        # Explain away from user k
                        explain = 1.0 - self.explain_strength * posteriors[k] + self.explain_strength / Q
                        new_channel *= explain

                        # Also explain away from other users that have run
                        for k3 in range(k):
                            if k3 != k2:
                                explain3 = 1.0 - self.explain_strength * posteriors[k3] + self.explain_strength / Q
                                new_channel *= explain3

                        # Normalize and store as log
                        row_sums = new_channel.sum(axis=-1, keepdims=True)
                        row_sums = np.where(row_sums < eps, 1.0, row_sums)
                        new_channel = new_channel / row_sums
                        channel[k2] = np.log(new_channel + eps)

            # Check convergence: all users pass syndrome
            hard = np.argmax(posteriors, axis=-1)  # [K, N]
            all_valid = all(self._check_syndrome(hard[k]) for k in range(K))

            # Also check that codewords are different
            if all_valid:
                unique_codewords = set(tuple(hard[k]) for k in range(K))
                if len(unique_codewords) == K:
                    break

        # Final hard decisions
        codewords = np.argmax(posteriors, axis=-1)  # [K, N]

        return codewords, posteriors

    def _normalize_prob_batch(self, p: np.ndarray) -> np.ndarray:
        """Normalize probability distributions along last axis. p: [..., Q]"""
        p = np.maximum(p, 0.0)
        s = p.sum(axis=-1, keepdims=True)
        s = np.where(s < self.eps, 1.0, s)
        return p / s

    def _normalize_log_batch(self, logp: np.ndarray) -> np.ndarray:
        """Convert log probs to normalized probs along last axis. logp: [..., Q]"""
        logp = logp.astype(float)
        m = logp.max(axis=-1, keepdims=True)
        p = np.exp(logp - m)
        s = p.sum(axis=-1, keepdims=True)
        s = np.where(s < self.eps, 1.0, s)
        return p / s

    def _bp_iteration_batch(
        self,
        channel_prob: np.ndarray,  # [B, N, Q]
        v2c: np.ndarray,           # [B, N, M, Q]
        c2v: np.ndarray,           # [B, M, N, Q]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One iteration of LDPC BP, batched over B samples.

        Returns:
            v2c_new [B, N, M, Q], c2v_new [B, M, N, Q], posteriors [B, N, Q]
        """
        B = channel_prob.shape[0]
        Q, N, M = self.Q, self.N, self.M
        damp = self.damping
        eps = self.eps

        # === Variable to check messages ===
        v2c_new = np.zeros_like(v2c)
        for n in range(N):
            checks = self.graph.var_to_checks[n]
            # Product of channel prob and all incoming check msgs: [B, Q]
            prod = channel_prob[:, n, :].copy()  # [B, Q]
            for m in checks:
                prod *= np.maximum(c2v[:, m, n, :], eps)  # [B, Q]
            prod = self._normalize_prob_batch(prod)

            # Extrinsic: exclude each check's own message
            for m in checks:
                msg = prod / np.maximum(c2v[:, m, n, :], eps)  # [B, Q]
                msg = self._normalize_prob_batch(msg)
                v2c_new[:, n, m, :] = (1 - damp) * v2c[:, n, m, :] + damp * msg

        # === Check to variable messages (WHT-based, batched) ===
        c2v_new = np.zeros_like(c2v)
        for m in range(M):
            vars_in_check = self.graph.check_to_vars[m]
            coeffs = self.graph.check_coeffs[m]
            d_c = len(vars_in_check)

            # Gather all incoming v2c messages for this check: [B, d_c, Q]
            v2c_gathered = np.stack(
                [v2c_new[:, v, m, :] for v in vars_in_check], axis=1
            )  # [B, d_c, Q]

            # GF-permute each edge: p_z[z] = p_x[h_inv * z]
            transformed = np.zeros_like(v2c_gathered)  # [B, d_c, Q]
            for i, h in enumerate(coeffs):
                h_inv = self._inv_table[h]
                perm = self._perm_table[h_inv]  # [Q]
                transformed[:, i, :] = v2c_gathered[:, i, :][:, perm]  # [B, Q]

            # WHT of each edge message: [B, d_c, Q]
            wht_all = _walsh_hadamard_transform(transformed)  # [..., Q] works

            # For each target edge, multiply WHTs of all OTHER edges (extrinsic)
            # Product of all WHTs: [B, Q]
            wht_prod = wht_all.prod(axis=1)  # [B, Q]

            for i, (v, h) in enumerate(zip(vars_in_check, coeffs)):
                # Extrinsic: divide out this edge's WHT
                wht_ext = wht_prod / (wht_all[:, i, :] + 1e-30)  # [B, Q]
                # Inverse WHT
                conv = _walsh_hadamard_transform(wht_ext) / Q  # [B, Q]
                conv = np.maximum(conv, 0)
                conv = self._normalize_prob_batch(conv)
                # GF-permute back: out[x] = conv[h * x]
                perm_fwd = self._perm_table[h]  # [Q]
                out = conv[:, perm_fwd]  # [B, Q]
                c2v_new[:, m, v, :] = (1 - damp) * c2v[:, m, v, :] + damp * out

        # === Posteriors ===
        posteriors = channel_prob.copy()  # [B, N, Q]
        for n in range(N):
            for m in self.graph.var_to_checks[n]:
                posteriors[:, n, :] *= np.maximum(c2v_new[:, m, n, :], eps)
            posteriors[:, n, :] = self._normalize_prob_batch(posteriors[:, n, :])

        return v2c_new, c2v_new, posteriors

    def decode_batch(self, Y_batch: np.ndarray, early_exit: bool = True) -> np.ndarray:
        """
        Decode a batch of observations, vectorized over B.

        All B samples go through BP iterations simultaneously.
        Shapes: v2c [B, N, M, Q], c2v [B, M, N, Q], etc.

        Args:
            Y_batch: [B, N, Q] soft scores
            early_exit: If True, snapshot per-sample posteriors at convergence
                (all K syndromes valid + K unique codewords) and break out of
                the iteration loop once every sample in the batch has converged.

        Returns:
            codewords: [B, K, N] decoded codewords
        """
        B, N, Q = Y_batch.shape
        K, M = self.K, self.M
        eps = self.eps

        # Channel probability from Y (shared across users)
        Y_prob = self._normalize_log_batch(Y_batch)  # [B, N, Q]

        # Per-user state: channel evidence, messages, posteriors
        channel = np.tile(Y_batch[:, np.newaxis, :, :], (1, K, 1, 1))  # [B, K, N, Q]
        v2c = np.ones((B, K, N, M, Q)) / Q
        c2v = np.ones((B, K, M, N, Q)) / Q
        posteriors = np.tile(Y_prob[:, np.newaxis, :, :], (1, K, 1, 1))  # [B, K, N, Q]

        # Per-sample convergence tracking
        done = np.zeros(B, dtype=bool) if early_exit else None
        final_posteriors = posteriors.copy() if early_exit else None

        for iter_idx in range(self.max_iters):
            for k in range(K):
                # BP iteration for user k across all B samples
                channel_prob_k = self._normalize_log_batch(channel[:, k])  # [B, N, Q]
                v2c[:, k], c2v[:, k], posteriors[:, k] = self._bp_iteration_batch(
                    channel_prob_k, v2c[:, k], c2v[:, k]
                )

                # Explain away: update other users' channel evidence
                for k2 in range(K):
                    if k2 != k:
                        new_ch = Y_prob.copy()  # [B, N, Q]
                        # Explain from user k
                        explain = 1.0 - self.explain_strength * posteriors[:, k] + self.explain_strength / Q
                        new_ch *= explain
                        # Explain from earlier users
                        for k3 in range(k):
                            if k3 != k2:
                                explain3 = 1.0 - self.explain_strength * posteriors[:, k3] + self.explain_strength / Q
                                new_ch *= explain3
                        new_ch = self._normalize_prob_batch(new_ch)
                        channel[:, k2] = np.log(new_ch + eps)

            # Per-sample convergence check + snapshot
            if early_exit:
                hard = np.argmax(posteriors, axis=-1)  # [B, K, N]
                for b in range(B):
                    if done[b]:
                        continue
                    all_valid = all(
                        self._check_syndrome(hard[b, k]) for k in range(K)
                    )
                    if not all_valid:
                        continue
                    if len(set(tuple(hard[b, k]) for k in range(K))) == K:
                        done[b] = True
                        final_posteriors[b] = posteriors[b]
                if done.all():
                    break

        if early_exit:
            # Use snapshots for samples that converged; current state for the rest.
            mask = done[:, None, None, None]
            posteriors = np.where(mask, final_posteriors, posteriors)

        codewords = np.argmax(posteriors, axis=-1)  # [B, K, N]
        return codewords


class FactorizedBPDemixer:
    """PyTorch-compatible wrapper for factorized BP decoder."""

    def __init__(self, config):
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))
        self.M = config.data.M

        # BP parameters
        self.max_iters = config.model.get('max_iters', 50)
        self.damping = config.model.get('damping', 0.5)
        self.explain_strength = config.model.get('explain_strength', 0.9)

        self.decoder = None
        self.H_matrix = None

    def set_H_matrix(self, H):
        """Set parity check matrix and initialize decoder."""
        import torch
        if isinstance(H, torch.Tensor):
            H = H.cpu().numpy()
        self.H_matrix = H
        self.decoder = FactorizedBPDecoder(
            Q=self.Q,
            N=self.N,
            K=self.K,
            M=self.M,
            H=H,
            max_iters=self.max_iters,
            damping=self.damping,
            explain_strength=self.explain_strength,
        )

    def forward(self, Y):
        """Decode codewords from observation."""
        import torch
        assert self.decoder is not None, "Must call set_H_matrix first"

        Y_np = Y.cpu().numpy()
        codewords = self.decoder.decode_batch(Y_np)

        return torch.from_numpy(codewords).to(Y.device)

    def __call__(self, Y):
        return self.forward(Y)
