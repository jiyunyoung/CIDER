"""
Beam-search Top-J list decoder (CPU or GPU), replacing the L^N brute-force
enumeration in rules/top_j_exhaustive_search.py.

This is the batched GPU Top-J search timed in the paper's main Table 1
(tab:main_results_classical, "Top-J (Top 2)" row) via
inference/bench_es_exhaustive.py, which runs it EXHAUSTIVELY (beam wide enough
that nothing is pruned). Naming: inside this module `L` is the proposal width
(the paper's J, top-J symbols per position) and `N` is the blocklength (the
paper's L).

The existing BeamSearchDecoder.decode() is named for beam search but actually
enumerates every one of L^N candidate codewords in a Python loop
(top_j_exhaustive_search.py:235), which is why the original-protocol table
(tab:app_classical_topj_results) reports DNF beyond L=18. Its beam machinery
(`beam_width`, `parity_impossible`, `Path`) is never called. This module
implements the algorithm that file's docstring describes.

Algorithm (per sample), matching the two-stage structure of the original:
  Stage 1 - find valid single codewords by beam search over positions:
      state = (symbols so far, running loglik, partial syndrome [M])
      at position i: expand each of B paths by the L top-L candidates,
      add log-evidence, XOR the GF contribution into the syndrome, kill any
      path whose *finalized* checks are already nonzero (the pruning rule of
      parity_impossible), then keep the top-B by loglik.
  Stage 2 - pick the K codewords that best explain the evidence, using the
      same coverage-then-loglik criterion as the original decoder.

Cost is O(N * B * L) instead of O(L^N) -- linear in blocklength.

Everything is batched over (samples x beam) as dense tensors, so the same
code runs on GPU: per position it issues a handful of large kernels rather
than one Python iteration per candidate.

Usage:
    dec = BeamTopJDecoder(Q, N, K, M, H, beam_width=512, proposal_width=2,
                          device='cuda')
    codewords = dec.decode_batch(Y)          # [S, K, N] numpy

    # Exhaustive Top-J (Table 1): beam_width = J**(N - s), seq_split = s,
    # i.e. J**s sequential passes that each keep every path of their subtree.
    dec = BeamTopJDecoder(Q, N, K, M, H, beam_width=J ** (N - s),
                          proposal_width=J, device='cuda', seq_split=s)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from rules.sic_bp import FactorizedBPDecoder


class BeamTopJDecoder:
    def __init__(self, Q: int, N: int, K: int, M: int, H: np.ndarray,
                 beam_width: int = 512, proposal_width: int = 2,
                 device: str = 'cpu', max_pairs: int = 64,
                 chunk: int = 256, seq_split: int = 0):
        self.Q, self.N, self.K, self.M = Q, N, K, M
        self.B = beam_width
        self.L = proposal_width
        self.max_pairs = max_pairs
        self.chunk = chunk
        # seq_split > 0: fix the top-L choice at the first `seq_split` positions
        # and run the beam over the remaining positions once per prefix,
        # SEQUENTIALLY (L**seq_split passes). Peak memory drops ~L**seq_split x
        # because only one prefix subtree is resident at a time. Output is
        # identical to the parallel path (exhaustive is exhaustive).
        self.seq_split = seq_split
        d = self.dev = torch.device(device)

        H = np.asarray(H, dtype=np.int64)
        self.H = torch.as_tensor(H, dtype=torch.long, device=d)          # [M,N]

        # GF multiply table, reused from the BP decoder's construction.
        ref = FactorizedBPDecoder(Q=Q, N=N, K=K, M=M, H=H)
        self.mul = torch.as_tensor(ref._mul_table, dtype=torch.long, device=d)

        # For each position, which checks become fully determined once that
        # position is assigned (the parity_impossible pruning rule).
        last_pos_of_check = [int(np.max(np.nonzero(H[m])[0])) for m in range(M)]
        self.finalized_at: List[List[int]] = [[] for _ in range(N)]
        for m, p in enumerate(last_pos_of_check):
            self.finalized_at[p].append(m)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _valid_codewords(self, Y: torch.Tensor):
        """Beam search over positions.

        Y: [S, N, Q] log-evidence.
        Returns syms [S, B, N] and loglik [S, B] with -inf for invalid paths.
        """
        S, N, Q = Y.shape
        B, L, M = self.B, self.L, self.M
        d = self.dev
        NEG = torch.finfo(Y.dtype).min / 4

        # top-L candidate symbols per (sample, position)
        cand_scores, cand_idx = torch.topk(Y, L, dim=-1)      # [S,N,L]

        syms = torch.zeros(S, 1, N, dtype=torch.long, device=d)
        loglik = torch.zeros(S, 1, dtype=Y.dtype, device=d)
        synd = torch.zeros(S, 1, M, dtype=torch.long, device=d)

        for i in range(N):
            c_idx = cand_idx[:, i, :]                          # [S,L]
            c_scr = cand_scores[:, i, :]                       # [S,L]
            Bcur = syms.shape[1]

            # GF contribution of each candidate to every check at position i
            h_col = self.H[:, i]                               # [M]
            contrib = self.mul[h_col.unsqueeze(0).unsqueeze(0),
                               c_idx.unsqueeze(-1)]            # [S,L,M]

            new_ll = loglik.unsqueeze(-1) + c_scr.unsqueeze(1)          # [S,B,L]
            new_sy = synd.unsqueeze(2) ^ contrib.unsqueeze(1)           # [S,B,L,M]

            # Kill paths whose finalized checks are already nonzero.
            fin = self.finalized_at[i]
            if fin:
                bad = (new_sy[..., fin] != 0).any(dim=-1)               # [S,B,L]
                new_ll = torch.where(bad, torch.full_like(new_ll, NEG), new_ll)

            flat_ll = new_ll.reshape(S, Bcur * L)
            keep = min(B, Bcur * L)
            top_ll, top_ix = torch.topk(flat_ll, keep, dim=1)           # [S,keep]

            b_ix, l_ix = top_ix // L, top_ix % L
            syms = torch.gather(
                syms, 1, b_ix.unsqueeze(-1).expand(-1, -1, N)).clone()
            syms[:, :, i] = torch.gather(c_idx, 1, l_ix)
            synd = new_sy.reshape(S, Bcur * L, M).gather(
                1, top_ix.unsqueeze(-1).expand(-1, -1, M))
            loglik = top_ll

        # Only all-zero-syndrome paths are codewords.
        ok = (synd == 0).all(dim=-1)                                    # [S,B]
        loglik = torch.where(ok, loglik, torch.full_like(loglik, NEG))
        return syms, loglik

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _valid_codewords_seq(self, Y: torch.Tensor, split: int):
        """Same exhaustive top-L enumeration as `_valid_codewords`, but run in
        L**split SEQUENTIAL passes instead of one parallel sweep.

        Each pass fixes the top-L choice at the first `split` positions to one
        prefix and beams over the remaining N-split positions, so at most one
        prefix subtree (L**(N-split) paths) is resident at a time. Across all
        prefixes this visits every one of the L**N leaves -- exhaustive.

        Only the best `max_pairs` parity-valid codewords per sample are carried
        between passes (that is all Stage 2 consumes), so the running buffer is
        tiny. Returns syms [S,V,N], loglik [S,V] with V<=max_pairs.
        """
        S, N, Q = Y.shape
        L, M = self.L, self.M
        d = self.dev
        NEG = torch.finfo(Y.dtype).min / 4
        keep_valid = self.max_pairs

        cand_scores, cand_idx = torch.topk(Y, L, dim=-1)          # [S,N,L]

        buf_sy = torch.zeros(S, 0, N, dtype=torch.long, device=d)
        buf_ll = torch.zeros(S, 0, dtype=Y.dtype, device=d)

        for pfx in range(L ** split):
            # prefix -> per-position choice index in [0,L)
            choices, x = [], pfx
            for _ in range(split):
                choices.append(x % L); x //= L

            syms = torch.zeros(S, 1, N, dtype=torch.long, device=d)
            loglik = torch.zeros(S, 1, dtype=Y.dtype, device=d)
            synd = torch.zeros(S, 1, M, dtype=torch.long, device=d)
            dead = torch.zeros(S, 1, dtype=torch.bool, device=d)

            # --- lay down the fixed prefix (Bcur stays 1) ---
            for i in range(split):
                ci = choices[i]
                sym_i = cand_idx[:, i, ci]                         # [S]
                syms[:, 0, i] = sym_i
                loglik[:, 0] = loglik[:, 0] + cand_scores[:, i, ci]
                contrib = self.mul[self.H[:, i].unsqueeze(0),
                                   sym_i.unsqueeze(1)]             # [S,M]
                synd[:, 0, :] = synd[:, 0, :] ^ contrib
                fin = self.finalized_at[i]
                if fin:
                    dead[:, 0] |= (synd[:, 0, fin] != 0).any(-1)
            loglik = torch.where(dead, torch.full_like(loglik, NEG), loglik)

            # --- beam over the remaining positions (exhaustive subtree) ---
            for i in range(split, N):
                c_idx = cand_idx[:, i, :]                          # [S,L]
                c_scr = cand_scores[:, i, :]                       # [S,L]
                Bcur = syms.shape[1]
                contrib = self.mul[self.H[:, i].unsqueeze(0).unsqueeze(0),
                                   c_idx.unsqueeze(-1)]            # [S,L,M]

                new_ll = loglik.unsqueeze(-1) + c_scr.unsqueeze(1)        # [S,B,L]
                new_sy = synd.unsqueeze(2) ^ contrib.unsqueeze(1)        # [S,B,L,M]
                fin = self.finalized_at[i]
                if fin:
                    bad = (new_sy[..., fin] != 0).any(dim=-1)
                    new_ll = torch.where(bad, torch.full_like(new_ll, NEG),
                                         new_ll)

                flat_ll = new_ll.reshape(S, Bcur * L)
                keep = min(L ** (N - split), Bcur * L)      # never truncate
                top_ll, top_ix = torch.topk(flat_ll, keep, dim=1)
                b_ix, l_ix = top_ix // L, top_ix % L
                syms = torch.gather(
                    syms, 1, b_ix.unsqueeze(-1).expand(-1, -1, N)).clone()
                syms[:, :, i] = torch.gather(c_idx, 1, l_ix)
                synd = new_sy.reshape(S, Bcur * L, M).gather(
                    1, top_ix.unsqueeze(-1).expand(-1, -1, M))
                loglik = top_ll

            # --- harvest this subtree's best valid leaves, merge into buffer ---
            ok = (synd == 0).all(dim=-1)                          # [S,Bsub]
            ll_masked = torch.where(ok, loglik,
                                    torch.full_like(loglik, NEG))
            kv = min(keep_valid, ll_masked.shape[1])
            tll, tix = torch.topk(ll_masked, kv, dim=1)
            tsy = torch.gather(syms, 1, tix.unsqueeze(-1).expand(-1, -1, N))
            buf_ll = torch.cat([buf_ll, tll], dim=1)
            buf_sy = torch.cat([buf_sy, tsy], dim=1)
            kk = min(keep_valid, buf_ll.shape[1])
            buf_ll, mix = torch.topk(buf_ll, kk, dim=1)
            buf_sy = torch.gather(buf_sy, 1, mix.unsqueeze(-1).expand(-1, -1, N))

        return buf_sy, buf_ll

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_batch(self, Y_batch, as_numpy: bool = True):
        """Y_batch: [S,N,Q] -> codewords [S,K,N] (numpy, or a device tensor if as_numpy=False)."""
        if isinstance(Y_batch, np.ndarray):
            Y = torch.as_tensor(Y_batch, dtype=torch.float32, device=self.dev)
        else:
            Y = Y_batch.to(self.dev, torch.float32)
        S, N, Q = Y.shape
        K = self.K
        NEG = torch.finfo(Y.dtype).min / 4

        if self.seq_split > 0:
            syms, loglik = self._valid_codewords_seq(Y, self.seq_split)
        else:
            syms, loglik = self._valid_codewords(Y)
        top1 = torch.argmax(Y, dim=-1)                                   # [S,N]
        topL = torch.topk(Y, self.L, dim=-1).indices                     # [S,N,L]

        # Keep the best few valid codewords per sample for the K-subset search.
        V = min(self.max_pairs, syms.shape[1])
        sel = torch.topk(loglik, V, dim=1).indices                       # [S,V]
        cand = torch.gather(sel.unsqueeze(-1).expand(-1, -1, N), 1,
                            torch.arange(V, device=self.dev)
                            .view(1, V, 1).expand(S, -1, N))
        cw = torch.gather(syms, 1, sel.unsqueeze(-1).expand(-1, -1, N))   # [S,V,N]
        ll = torch.gather(loglik, 1, sel)                                 # [S,V]
        valid = ll > NEG / 2

        # Stage 2, fully vectorized over samples. A per-sample Python loop here
        # costs ~0.73 ms/sample on GPU (99% of runtime) because it issues a few
        # tiny kernels per sample; batching it collapses that to a handful of
        # large kernels. Chunked over S to bound the [S,P,N,L] intermediate.
        out = torch.zeros(S, K, N, dtype=torch.long, device=self.dev)
        nvalid = valid.sum(dim=1)                                        # [S]
        Vp = cw.shape[1]

        if K == 2 and Vp >= 2:
            ii, jj = torch.triu_indices(Vp, Vp, offset=1, device=self.dev)
            for lo in range(0, S, self.chunk):
                hi = min(lo + self.chunk, S)
                a = cw[lo:hi][:, ii, :]                                  # [c,P,N]
                b = cw[lo:hi][:, jj, :]
                tl = topL[lo:hi].unsqueeze(1)                            # [c,1,N,L]
                in_a = (a.unsqueeze(-1) == tl).any(-1)                   # [c,P,N]
                in_b = (b.unsqueeze(-1) == tl).any(-1)
                pair_hits = in_a & in_b & (a != b)
                collide = (a == b) & (a == top1[lo:hi].unsqueeze(1))
                coverage = (pair_hits | collide).sum(-1)                 # [c,P]
                pair_ok = valid[lo:hi][:, ii] & valid[lo:hi][:, jj]
                sc = coverage.to(ll.dtype) * 1e6 + (ll[lo:hi][:, ii] +
                                                    ll[lo:hi][:, jj])
                sc = torch.where(pair_ok, sc, torch.full_like(sc, NEG))
                best = sc.argmax(dim=1)                                  # [c]
                gid = best.view(-1, 1, 1).expand(-1, 1, N)
                out[lo:hi, 0] = a.gather(1, gid).squeeze(1)
                out[lo:hi, 1] = b.gather(1, gid).squeeze(1)
        else:
            kk = min(K, Vp)
            idx = torch.topk(ll, kk, dim=1).indices                      # [S,kk]
            out[:, :kk] = torch.gather(
                cw, 1, idx.unsqueeze(-1).expand(-1, -1, N))

        # Fallbacks: no valid codeword -> top-1 repeated; fewer than K -> repeat
        # the best valid one. Mirrors the original decoder's behaviour.
        none = nvalid == 0
        if none.any():
            out[none] = top1[none].unsqueeze(1).expand(-1, K, -1)
        few = (nvalid > 0) & (nvalid < K)
        if few.any():
            first = torch.argmax(valid.to(torch.uint8), dim=1)           # [S]
            pick = torch.gather(
                cw, 1, first.view(-1, 1, 1).expand(-1, 1, N)).squeeze(1)
            out[few] = pick[few].unsqueeze(1).expand(-1, K, -1)
        return out.cpu().numpy() if as_numpy else out
