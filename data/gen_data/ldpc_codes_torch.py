"""
q-ary LDPC code implementation using PEG (Progressive Edge Growth) construction.
PyTorch version for neural network training.

This module provides efficient q-ary LDPC code generation and encoding
using the Progressive Edge Growth algorithm and systematic form encoding.
Returns codewords and matrices as PyTorch tensors for GPU acceleration.
"""

from typing import List, Tuple
from collections import deque
import numpy as np
import galois
import torch


# =============================================================================
# Base class for outer codes
# =============================================================================

class OuterCodec:
    """
    Base class for outer codes.

    An outer code maps messages m in [0, M) to codewords c in [0, Q)^L.
    """

    def __init__(self, code_len: int, code_dim: int):
        self.code_len = code_len
        self.code_dim = code_dim
        self.M = 1 << code_dim  # 2^code_dim
        self.codebook = None

    def encode(self, message: int) -> List[int]:
        raise NotImplementedError

    def get_codebook(self) -> List[List[int]]:
        if self.codebook is None:
            self._generate_codebook()
        return self.codebook

    def _generate_codebook(self):
        raise NotImplementedError


# =============================================================================
# LDPC Code Functions
# =============================================================================

def bfs_distance(v_start, c_target, var_to_chk, chk_to_var):
    """
    BFS on Tanner graph from variable v_start.

    Searching distance to check c_target.
    Distance counts edges: v→c→v→c...

    Args:
        v_start: Starting variable node index
        c_target: Target check node index
        var_to_chk: List of check nodes connected to each variable node
        chk_to_var: List of variable nodes connected to each check node

    Returns:
        Distance (number of edges) to reach c_target from v_start, or -1 if unreachable
    """
    queue = deque([(v_start, 0)])  # (variable_node, distance)
    visited_v = {v_start}
    visited_c = set()

    while queue:
        v, d = queue.popleft()

        # Try to reach target check
        for c in var_to_chk[v]:
            if c == c_target:
                return d + 1

            if c not in visited_c:
                visited_c.add(c)

                # Expand to variable neighbors
                for v2 in chk_to_var[c]:
                    if v2 not in visited_v:
                        visited_v.add(v2)
                        queue.append((v2, d + 2))

    # unreachable → return -1 (smallest distance)
    return -1


def _gcd(a, b):
    """Compute GCD of two integers."""
    while b:
        a, b = b, a % b
    return a


def _build_base_protograph(m_b, n_b, d_v, d_c):
    """
    Build a connected base protograph matrix B with uniform mixing.

    Prefers structures where each check type connects to all variable types
    for uniform syndrome information flow.

    Args:
        m_b: Number of check node types (rows)
        n_b: Number of variable node types (columns)
        d_v: Variable node degree (column sum)
        d_c: Check node degree (row sum)

    Returns:
        B: Base matrix (m_b × n_b) with edge multiplicities
    """
    # Verify constraint: n_b * d_v = m_b * d_c
    assert n_b * d_v == m_b * d_c, f"Degree mismatch: {n_b}*{d_v} != {m_b}*{d_c}"

    total_edges = n_b * d_v  # = m_b * d_c

    # Strategy: Spread edges as uniformly as possible across all (i,j) pairs
    # This ensures each check type connects to all variable types
    B = np.zeros((m_b, n_b), dtype=int)

    # Base value for each entry
    base = total_edges // (m_b * n_b)
    remainder = total_edges % (m_b * n_b)

    # Fill with base value
    B[:, :] = base

    # Distribute remainder evenly using diagonal pattern
    count = 0
    for offset in range(max(m_b, n_b)):
        for i in range(m_b):
            j = (i + offset) % n_b
            if count < remainder:
                B[i, j] += 1
                count += 1

    # Verify and fix if needed
    col_sums = B.sum(axis=0)
    row_sums = B.sum(axis=1)

    # Adjust if constraints not perfectly met
    while not (np.all(col_sums == d_v) and np.all(row_sums == d_c)):
        # Find a cell to decrement (row has excess, col has excess)
        for i in range(m_b):
            for j in range(n_b):
                if row_sums[i] > d_c and col_sums[j] > d_v and B[i, j] > 0:
                    B[i, j] -= 1
                    row_sums[i] -= 1
                    col_sums[j] -= 1
                    break
            else:
                continue
            break
        else:
            # Find a cell to increment
            for i in range(m_b):
                for j in range(n_b):
                    if row_sums[i] < d_c and col_sums[j] < d_v:
                        B[i, j] += 1
                        row_sums[i] += 1
                        col_sums[j] += 1
                        break
                else:
                    continue
                break
            else:
                break  # No changes possible

        col_sums = B.sum(axis=0)
        row_sums = B.sum(axis=1)

    assert np.all(B.sum(axis=0) == d_v), f"Column sums: {B.sum(axis=0)}, expected {d_v}"
    assert np.all(B.sum(axis=1) == d_c), f"Row sums: {B.sum(axis=1)}, expected {d_c}"

    return B


def _is_protograph_connected(B):
    """Check if protograph base matrix represents a connected bipartite graph."""
    m_b, n_b = B.shape

    # Build adjacency for BFS (variable nodes: 0..n_b-1, check nodes: n_b..n_b+m_b-1)
    visited = [False] * (n_b + m_b)
    queue = [0]  # Start from first variable node
    visited[0] = True

    while queue:
        node = queue.pop(0)
        if node < n_b:  # Variable node
            j = node
            for i in range(m_b):
                if B[i, j] > 0:
                    check_node = n_b + i
                    if not visited[check_node]:
                        visited[check_node] = True
                        queue.append(check_node)
        else:  # Check node
            i = node - n_b
            for j in range(n_b):
                if B[i, j] > 0:
                    if not visited[j]:
                        visited[j] = True
                        queue.append(j)

    return all(visited)


def _compute_local_girth(var_to_chk, chk_to_var, v_start, max_depth=12):
    """
    Compute the local girth (shortest cycle) starting from variable node v_start.
    Returns the length of the shortest cycle, or max_depth+1 if none found.
    """
    # BFS to find shortest cycle through v_start
    # A cycle is found when we return to v_start
    L = len(var_to_chk)
    M = len(chk_to_var)

    # Track (node_type, node_idx, depth, parent)
    # node_type: 'v' for variable, 'c' for check
    from collections import deque

    queue = deque()
    # Start from v_start, go to its check neighbors
    for c in var_to_chk[v_start]:
        queue.append(('c', c, 1, ('v', v_start)))

    visited = {}  # (type, idx) -> depth at first visit

    while queue:
        node_type, node_idx, depth, parent = queue.popleft()

        if depth > max_depth:
            continue

        key = (node_type, node_idx)
        if key in visited:
            continue
        visited[key] = depth

        if node_type == 'c':
            # Expand to variable neighbors
            for v in chk_to_var[node_idx]:
                if v == v_start and depth >= 3:
                    # Found a cycle back to start
                    return depth + 1
                if ('v', v) not in visited and (parent != ('v', v)):
                    queue.append(('v', v, depth + 1, ('c', node_idx)))
        else:  # variable node
            for c in var_to_chk[node_idx]:
                if ('c', c) not in visited and (parent != ('c', c)):
                    queue.append(('c', c, depth + 1, ('v', node_idx)))

    return max_depth + 1


def _select_circulant_shifts_peg(B, Z, seed=None):
    """
    Select circulant shift values using PEG-like algorithm to maximize girth.

    For each edge in the protograph, try all candidate shifts and pick
    the one that maximizes the local girth in the lifted graph.

    Args:
        B: Base protograph matrix (m_b × n_b)
        Z: Lifting factor
        seed: Random seed

    Returns:
        S: List matrix (same shape as B), each entry is a list of shifts
    """
    if seed is not None:
        np.random.seed(seed)

    m_b, n_b = B.shape
    L = n_b * Z  # Total variable nodes
    M = m_b * Z  # Total check nodes

    # S[i][j] is a list of shifts for that block
    S = [[[] for _ in range(n_b)] for _ in range(m_b)]

    # Current graph state
    var_to_chk = [[] for _ in range(L)]
    chk_to_var = [[] for _ in range(M)]

    # Collect all edges to place: (i, j, edge_idx within block)
    edges_to_place = []
    for i in range(m_b):
        for j in range(n_b):
            for e in range(B[i, j]):
                edges_to_place.append((i, j, e))

    # Ensure first edge uses shift coprime with Z for connectivity
    first_edge_placed = False

    for i, j, e in edges_to_place:
        best_shift = 0
        best_girth = -1

        # Candidate shifts
        if not first_edge_placed and Z > 1:
            # First edge: prefer shift = 1 (coprime with any Z)
            candidates = [1] + [s for s in range(Z) if s != 1]
        else:
            candidates = list(range(Z))

        # Try each candidate shift
        for shift in candidates:
            # Check if this shift is already used in this block
            if shift in S[i][j]:
                continue

            # Temporarily add edges for this shift
            temp_edges = []
            for z in range(Z):
                check_idx = i * Z + z
                var_idx = j * Z + (z + shift) % Z
                var_to_chk[var_idx].append(check_idx)
                chk_to_var[check_idx].append(var_idx)
                temp_edges.append((var_idx, check_idx))

            # Compute local girth for affected nodes
            min_girth = float('inf')
            for var_idx, check_idx in temp_edges[:min(4, Z)]:  # Sample a few nodes
                girth = _compute_local_girth(var_to_chk, chk_to_var, var_idx)
                min_girth = min(min_girth, girth)

            # Remove temporary edges
            for var_idx, check_idx in temp_edges:
                var_to_chk[var_idx].pop()
                chk_to_var[check_idx].pop()

            # Track best shift
            if min_girth > best_girth:
                best_girth = min_girth
                best_shift = shift

        # Apply best shift
        S[i][j].append(best_shift)
        for z in range(Z):
            check_idx = i * Z + z
            var_idx = j * Z + (z + best_shift) % Z
            var_to_chk[var_idx].append(check_idx)
            chk_to_var[check_idx].append(var_idx)

        first_edge_placed = True

    return S


def protograph_ldpc_qary(q, L, M, d_v, d_c, seed=None):
    """
    Generate q-ary QC-LDPC using protograph-based construction.

    This approach guarantees connectivity by:
    1. Building a connected base protograph with uniform mixing
    2. Lifting with circulant permutations (Z = GCD(L, M))
    3. PEG-based shift selection to maximize girth

    Args:
        q: Alphabet size (power of 2 for GF(q))
        L: Code length (number of variable nodes)
        M: Number of parity checks (check nodes)
        d_v: Variable node degree
        d_c: Check node degree
        seed: Random seed

    Returns:
        H: Parity check matrix in GF(q) with shape (M, L)
        var_to_chk: List of check nodes connected to each variable node
        chk_to_var: List of variable nodes connected to each check node
    """
    if seed is not None:
        np.random.seed(seed)

    GF = galois.GF(q)

    assert L * d_v == M * d_c, f"Degree condition L*d_v = M*d_c must hold. Got L={L}, d_v={d_v}, M={M}, d_c={d_c}"

    # Step 1: Compute lifting factor Z = GCD(L, M)
    Z = _gcd(L, M)
    n_b = L // Z  # Variable node types in protograph
    m_b = M // Z  # Check node types in protograph

    # Step 2: Build connected base protograph
    B = _build_base_protograph(m_b, n_b, d_v, d_c)

    if not _is_protograph_connected(B):
        raise ValueError("Failed to build connected protograph")

    # Step 3: Select circulant shifts using PEG (maximize girth)
    S = _select_circulant_shifts_peg(B, Z, seed)

    # Step 4: Build full H matrix by QC lifting
    H = GF.Zeros((M, L))
    var_to_chk = [[] for _ in range(L)]
    chk_to_var = [[] for _ in range(M)]

    for i in range(m_b):  # For each check type
        for j in range(n_b):  # For each variable type
            shifts = S[i][j]  # List of shifts for this block
            for shift in shifts:
                # Place Z×Z circulant block at position (i*Z, j*Z)
                for z in range(Z):
                    # Circulant: row z connects to column (z + shift) % Z
                    check_idx = i * Z + z
                    var_idx = j * Z + (z + shift) % Z

                    # Random nonzero GF(q) label
                    H[check_idx, var_idx] = GF.Random(low=1)

                    # Update adjacency lists
                    var_to_chk[var_idx].append(check_idx)
                    chk_to_var[check_idx].append(var_idx)

    return H, var_to_chk, chk_to_var


def peg_ldpc_qary(q, L, M, d_v, d_c, seed=None):
    """
    Generate q-ary LDPC code. Uses protograph-based QC construction.

    This is a wrapper that calls protograph_ldpc_qary for guaranteed
    connectivity, falling back to PEG-based construction if needed.

    Args:
        q: Alphabet size (must be power of 2 for GF(q))
        L: Code length (number of variable nodes)
        M: Number of parity checks (check nodes)
        d_v: Variable node degree
        d_c: Check node degree
        seed: Random seed for reproducibility

    Returns:
        H: Parity check matrix in GF(q) with shape (M, L)
        var_to_chk: List of check nodes connected to each variable node
        chk_to_var: List of variable nodes connected to each check node
    """
    try:
        return protograph_ldpc_qary(q, L, M, d_v, d_c, seed)
    except (ValueError, AssertionError) as e:
        # Fall back to PEG-based construction if protograph fails
        return _peg_ldpc_qary_fallback(q, L, M, d_v, d_c, seed)


def _peg_ldpc_qary_fallback(q, L, M, d_v, d_c, seed=None):
    """
    Fallback PEG-based construction with connectivity enforcement.
    Used when protograph construction is not feasible.
    """
    if seed is not None:
        np.random.seed(seed)

    GF = galois.GF(q)

    assert L * d_v == M * d_c, f"Degree condition L*d_v = M*d_c must hold. Got L={L}, d_v={d_v}, M={M}, d_c={d_c}"

    var_to_chk = [[] for _ in range(L)]
    chk_to_var = [[] for _ in range(M)]

    # Phase 1: Build connected backbone
    for v in range(L):
        c = v % M
        var_to_chk[v].append(c)
        chk_to_var[c].append(v)

    # Phase 1b: Add bridging edges for connectivity
    for v in range(1, L):
        if len(var_to_chk[v]) >= d_v:
            continue
        for prev_v in range(v - 1, max(-1, v - M - 1), -1):
            connected = False
            for c in var_to_chk[prev_v]:
                if c not in var_to_chk[v] and len(chk_to_var[c]) < d_c:
                    var_to_chk[v].append(c)
                    chk_to_var[c].append(v)
                    connected = True
                    break
            if connected:
                break

    # Phase 2: Fill remaining edges using PEG
    for v in range(L):
        while len(var_to_chk[v]) < d_v:
            best_dist = -2
            candidates = []

            for c in range(M):
                if len(chk_to_var[c]) >= d_c:
                    continue
                if c in var_to_chk[v]:
                    continue

                dist = bfs_distance(v, c, var_to_chk, chk_to_var)
                if dist > best_dist:
                    best_dist = dist
                    candidates = [c]
                elif dist == best_dist:
                    candidates.append(c)

            if not candidates:
                break

            best_c = candidates[np.random.randint(len(candidates))]
            var_to_chk[v].append(best_c)
            chk_to_var[best_c].append(v)

    # Build q-ary H
    H = GF.Zeros((M, L))

    for c in range(M):
        for v in chk_to_var[c]:
            H[c, v] = GF.Random(low=1)

    return H, var_to_chk, chk_to_var


def _calculate_rank(matrix):
    """Calculate rank of a matrix in GF(q) by counting non-zero rows in row-reduced form."""
    R = matrix.row_reduce()
    rank = 0
    for i in range(R.shape[0]):
        if not all(R[i] == 0):
            rank += 1
    return rank


def make_systematic_from_H(H):
    """
    Convert parity-check matrix H into systematic form.

    Finds a column permutation Pi such that H[:, Pi] = [H1 | H2]
    where H2 is invertible, enabling systematic encoding.

    Args:
        H: Parity check matrix in GF(q) with shape (M, L)

    Returns:
        H1: First M columns part of H[: , Pi], shape (M, k)
        H2: Last M columns part of H[:, Pi], shape (M, M)
        Pi: Permutation array of column indices
        GF: Galois Field type
    """
    GF = type(H)
    M, L = H.shape

    # Find M linearly independent columns (these will become H2)
    # and the remaining columns (these will become H1)

    # Build H2 by selecting linearly independent columns
    H2_cols = []
    remaining_cols = list(range(L))

    for target_rank in range(1, M + 1):
        found = False
        for i, col_idx in enumerate(remaining_cols):
            # Try adding this column to H2
            test_H2_cols = H2_cols + [col_idx]
            test_H2 = H[:, test_H2_cols]

            # Check rank (all columns of test_H2 should be linearly independent)
            rank = _calculate_rank(test_H2)

            if rank == target_rank:
                H2_cols.append(col_idx)
                remaining_cols.pop(i)
                found = True
                break

        if not found:
            raise ValueError(f"Could not find {M} linearly independent columns in H")

    # H1 columns are the remaining ones
    H1_cols = remaining_cols
    k = len(H1_cols)

    # Permutation: information first, parity second
    Pi = H1_cols + H2_cols

    # Extract H1 and H2 from permuted H
    H1 = H[:, H1_cols]
    H2 = H[:, H2_cols]

    return H1, H2, Pi, GF


def ldpc_encode(H1, H2, H2_inv, Pi, GF, u):
    """
    Efficient q-ary LDPC encoder using systematic form.

    Given H = [H1 | H2] in systematic form and information vector u,
    computes parity symbols p = -H2^{-1} @ H1 @ u,
    and returns the full codeword with proper permutation.

    Args:
        H1: Information part of H, shape (M, k)
        H2: Parity part of H, shape (M, M) (must be invertible)
        H2_inv: Inverse of H2, shape (M, M)
        Pi: Permutation array
        GF: Galois Field type
        u: Information vector (length k), can be list or array

    Returns:
        v: Codeword vector in GF(q) of length L, satisfying H @ v = 0 (mod Q)
    """
    u = GF(u)

    # Compute parity: p = -(H2_inv @ H1 @ u)
    Hu = H1 @ u
    p = -(H2_inv @ Hu)

    # Form codeword in permuted coordinates: [u | p]
    v_perm = GF.Zeros(len(Pi))
    k = len(u)
    v_perm[:k] = u
    v_perm[k:] = p

    # Undo permutation to get codeword in original coordinates
    v = GF.Zeros(len(Pi))
    for i, col in enumerate(Pi):
        v[col] = v_perm[i]

    return v


class PEGLDPCOuterCodeTorch(OuterCodec):
    """
    q-ary LDPC code with PEG (Progressive Edge Growth) construction.
    PyTorch version - returns tensors instead of lists.

    Generates systematic q-ary LDPC codes using:
    1. PEG algorithm for Tanner graph construction
    2. Gaussian elimination for systematic form
    3. Efficient systematic encoding

    Properties:
    - Sparse parity-check matrix H in GF(q) with shape (M, L)
    - Variable node degree d_v and check node degree d_c
    - Degree condition: L*d_v = M*d_c
    - Systematic form: first k symbols are information, last m are parity
    - All outputs as PyTorch tensors for GPU acceleration
    """

    def __init__(self, q: int, code_len: int, code_rate: float,
                 d_v: int = 3, d_c: int = 5, seed: int = 0, device: str = 'cpu'):
        """
        Initialize q-ary LDPC code with PEG construction.

        Args:
            q: Alphabet size (GF(q), must be power of 2)
            code_len: Code length L (number of variable nodes)
            code_rate: Code rate R (determines M = L(1-R) check nodes)
            d_v: Variable node degree
            d_c: Check node degree (will be adjusted if needed)
            seed: Random seed for reproducibility
            device: 'cpu' or 'cuda' for tensor device

        Raises:
            ValueError: If parameters don't satisfy degree condition
            ValueError: If code_rate is invalid
        """
        self.q = q
        self.L = code_len
        self.R = code_rate
        self.device = device

        if code_rate <= 0 or code_rate >= 1:
            raise ValueError(f"Code rate must be in (0, 1), got {code_rate}")

        # Compute dimensions
        num_check_nodes = int((1 - code_rate) * code_len)  # Number of check nodes in Tanner graph
        k = code_len - num_check_nodes  # Information symbols

        if k < 1:
            raise ValueError(f"Information dimension k={k} must be >= 1. "
                           f"Reduce code_rate or increase code_len.")

        self.d_v = d_v
        # Adjust d_c if needed to satisfy degree condition
        if (code_len * d_v) % num_check_nodes != 0:
            # Find nearest valid d_c
            self.d_c = max(1, round((code_len * d_v) / num_check_nodes))
        else:
            self.d_c = d_c if (code_len * d_v) == (num_check_nodes * d_c) else \
                      (code_len * d_v) // num_check_nodes

        # Validate degree condition
        if (code_len * self.d_v) != (num_check_nodes * self.d_c):
            raise ValueError(f"Cannot satisfy degree condition L*d_v = M*d_c "
                           f"with L={code_len}, d_v={d_v}, num_checks={num_check_nodes}. "
                           f"Try different parameters.")

        # Initialize OuterCodec: use code_dim = 1 as placeholder since we handle q^k differently
        # For q-ary codes, M = q^k, not 2^k as in base class
        code_dim = 1
        super().__init__(code_len, code_dim)

        # Store LDPC-specific parameters
        self.num_check_nodes = num_check_nodes
        self.k = k
        # Override M to be q^k for q-ary codes
        self.M = q ** k

        # Initialize GF
        self.GF = galois.GF(q)

        # Generate LDPC structure
        np.random.seed(seed)
        self.H, self.var_to_chk, self.chk_to_var = peg_ldpc_qary(
            q, code_len, code_rate, self.d_v, self.d_c, seed=seed
        )

        # Convert to systematic form
        self.H1, self.H2, self.Pi, _ = make_systematic_from_H(self.H)

        # Compute H2^{-1} using numpy (galois doesn't have inv method)
        self.H2_inv = self.GF(np.linalg.inv(self.H2))

        # Convert H to torch tensor for easy access
        self._H_tensor = None
        self._H_torch_cached = False

        # Generate codebook only if reasonable size (to avoid memory issues)
        # For large q^k, encode() will compute on-the-fly instead
        self.CODEBOOK_THRESHOLD = 100000  # Don't pre-generate if M > 100k
        self.codebook = None
        if self.M <= self.CODEBOOK_THRESHOLD:
            self._generate_codebook()

    def _generate_codebook(self):
        """
        Generate all q^k codewords using efficient systematic encoding.

        For q-ary LDPC codes, iterates through all q^k possible information vectors.
        Stores codewords as torch tensors for efficient GPU transfer.
        """
        self.codebook = []

        for msg in range(self.M):
            # Convert msg from base-10 to base-q representation (k digits)
            u_list = []
            temp = msg
            for _ in range(self.k):
                u_list.append(temp % self.q)
                temp //= self.q

            # Create information vector in GF(q)
            u = self.GF(np.array(u_list, dtype=np.int32))

            # Encode using systematic form: p = -H2_inv @ H1 @ u
            v = ldpc_encode(self.H1, self.H2, self.H2_inv, self.Pi, self.GF, u)

            # Convert to torch tensor (on CPU first, can be moved to GPU as needed)
            codeword = torch.tensor([int(v[i]) for i in range(self.L)],
                                   dtype=torch.long, device='cpu')
            self.codebook.append(codeword)

    def encode(self, message: int) -> torch.Tensor:
        """
        Encode message into q-ary LDPC codeword.

        Args:
            message: Message index in [0, M)

        Returns:
            Codeword tensor of shape (L,) with values in [0, q), on specified device
        """
        msg = message % self.M

        # Use pre-generated codebook if available
        if self.codebook is not None:
            return self.codebook[msg].to(self.device)

        # Otherwise compute on-the-fly
        # Convert msg from base-10 to base-q representation (k digits)
        u_list = []
        temp = msg
        for _ in range(self.k):
            u_list.append(temp % self.q)
            temp //= self.q

        # Create information vector in GF(q)
        u = self.GF(np.array(u_list, dtype=np.int32))

        # Encode using systematic form: p = -H2_inv @ H1 @ u
        v = ldpc_encode(self.H1, self.H2, self.H2_inv, self.Pi, self.GF, u)

        # Convert to torch tensor
        codeword = torch.tensor([int(v[i]) for i in range(self.L)],
                               dtype=torch.long, device=self.device)
        return codeword

    def get_H_tensor(self, device: str = None) -> torch.Tensor:
        """
        Get parity-check matrix H as PyTorch tensor.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').
                   Uses self.device if None.

        Returns:
            H tensor of shape (num_check_nodes, L) with values in [0, q)
        """
        if device is None:
            device = self.device

        if not self._H_torch_cached:
            H_array = np.array(self.H, dtype=np.int32)
            self._H_tensor = torch.tensor(H_array, dtype=torch.long, device='cpu')
            self._H_torch_cached = True

        return self._H_tensor.to(device)

    def verify_codeword(self, codeword) -> bool:
        """
        Verify that a codeword satisfies the parity check constraint H*c = 0 (mod Q).

        Uses the original H matrix for verification.

        Args:
            codeword: Codeword to verify (list, array, or tensor of length L)

        Returns:
            True if H * c = 0 (mod Q), False otherwise
        """
        # Convert tensor to list if needed
        if isinstance(codeword, torch.Tensor):
            codeword = codeword.cpu().numpy().tolist()

        c_gf = self.GF(codeword)
        hc = self.H @ c_gf
        return all(hc[i] == 0 for i in range(len(hc)))

    def get_tanner_graph_stats(self) -> dict:
        """
        Return statistics about the Tanner graph structure.

        Returns:
            Dictionary with graph properties
        """
        total_edges = sum(len(neighbors) for neighbors in self.var_to_chk)
        avg_var_degree = total_edges / self.L if self.L > 0 else 0
        avg_check_degree = total_edges / self.num_check_nodes if self.num_check_nodes > 0 else 0

        return {
            'q': self.q,
            'L': self.L,
            'num_check_nodes': self.num_check_nodes,
            'codebook_size': self.M,
            'k': self.k,
            'rate': self.R,
            'n_variables': self.L,
            'n_checks': self.num_check_nodes,
            'total_edges': total_edges,
            'avg_variable_degree': avg_var_degree,
            'avg_check_degree': avg_check_degree,
            'target_d_v': self.d_v,
            'target_d_c': self.d_c,
        }
