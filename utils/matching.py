"""
Matching algorithms for permutation-invariant loss computation.

Hungarian: O(K³) - optimal but slow
Greedy: O(K²) - fast approximation, ~95-99% same as Hungarian
"""

import torch
from scipy.optimize import linear_sum_assignment


def hungarian_match(cost_matrix: torch.Tensor):
    """
    Optimal assignment using Hungarian algorithm.

    Args:
        cost_matrix: [K, K] cost matrix where cost[i,j] = cost of assigning pred_i to gt_j

    Returns:
        row_ind: [K] prediction indices
        col_ind: [K] ground truth indices (permutation)
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    return torch.tensor(row_ind), torch.tensor(col_ind)


def greedy_match(cost_matrix: torch.Tensor):
    """
    Fast greedy matching - O(K²) instead of O(K³).

    Iteratively picks the best remaining match.
    Not globally optimal but very fast and works well in practice.

    Args:
        cost_matrix: [K, K] cost matrix where cost[i,j] = cost of assigning pred_i to gt_j

    Returns:
        row_ind: [K] prediction indices
        col_ind: [K] ground truth indices (permutation)
    """
    K = cost_matrix.shape[0]
    device = cost_matrix.device

    cost = cost_matrix.clone()
    row_ind = []
    col_ind = []

    for _ in range(K):
        # Find minimum cost assignment
        flat_idx = cost.argmin()
        i, j = flat_idx // K, flat_idx % K

        row_ind.append(i.item())
        col_ind.append(j.item())

        # Mark row i and column j as used (set to inf)
        cost[i, :] = float('inf')
        cost[:, j] = float('inf')

    # Sort by row index to get proper order
    sorted_pairs = sorted(zip(row_ind, col_ind))
    row_ind = [p[0] for p in sorted_pairs]
    col_ind = [p[1] for p in sorted_pairs]

    return torch.tensor(row_ind), torch.tensor(col_ind)


def batched_greedy_match(cost_matrices: torch.Tensor):
    """
    Batched greedy matching for multiple samples.

    Args:
        cost_matrices: [B, K, K] batch of cost matrices

    Returns:
        col_indices: [B, K] permutation for each sample
    """
    B, K, _ = cost_matrices.shape
    device = cost_matrices.device

    col_indices = torch.zeros(B, K, dtype=torch.long, device=device)

    cost = cost_matrices.clone()

    for step in range(K):
        # Find minimum in each sample's remaining costs
        flat_cost = cost.view(B, -1)  # [B, K*K]
        flat_idx = flat_cost.argmin(dim=1)  # [B]

        row_idx = flat_idx // K  # [B]
        col_idx = flat_idx % K   # [B]

        # Store assignment
        for b in range(B):
            col_indices[b, row_idx[b]] = col_idx[b]

        # Mask used rows and columns
        for b in range(B):
            cost[b, row_idx[b], :] = float('inf')
            cost[b, :, col_idx[b]] = float('inf')

    return col_indices


def compute_cost_matrix(pred_logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
    """
    Compute cost matrix for matching.

    Args:
        pred_logits: [K, N, Q] logits
        gt: [K, N] ground truth symbols
        mask: [K, N] optional mask (1 = compute loss, 0 = ignore)

    Returns:
        cost: [K, K] cost matrix
    """
    K, N, Q = pred_logits.shape

    # Get predictions
    pred = pred_logits.argmax(dim=-1)  # [K, N]

    # Compute cost: number of mismatches between pred[i] and gt[j]
    cost = torch.zeros(K, K, device=pred_logits.device)

    for i in range(K):
        for j in range(K):
            if mask is not None:
                # Only count masked positions
                mismatch = ((pred[i] != gt[j]) & mask[j]).float().sum()
            else:
                mismatch = (pred[i] != gt[j]).float().sum()
            cost[i, j] = mismatch

    return cost


def compute_cost_matrix_batched(pred_logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
    """
    Batched cost matrix computation.

    Args:
        pred_logits: [B, K, N, Q] logits
        gt: [B, K, N] ground truth symbols
        mask: [B, K, N] optional mask

    Returns:
        cost: [B, K, K] cost matrices
    """
    B, K, N, Q = pred_logits.shape
    device = pred_logits.device

    # Get predictions
    pred = pred_logits.argmax(dim=-1)  # [B, K, N]

    # Expand for pairwise comparison: pred[b,i,n] vs gt[b,j,n]
    pred_exp = pred.unsqueeze(2)  # [B, K, 1, N]
    gt_exp = gt.unsqueeze(1)      # [B, 1, K, N]

    # Mismatch: [B, K, K, N]
    mismatch = (pred_exp != gt_exp).float()

    if mask is not None:
        # mask is [B, K, N], expand to [B, 1, K, N] for gt positions
        mask_exp = mask.unsqueeze(1)
        mismatch = mismatch * mask_exp

    # Sum over positions: [B, K, K]
    cost = mismatch.sum(dim=-1)

    return cost
