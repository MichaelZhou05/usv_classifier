"""Sliced-Wasserstein Embedding (SWE) Pooling.

A sophisticated pooling method based on optimal transport that creates
fixed-dimensional permutation-invariant embeddings for sets of arbitrary size.

Reference: https://arxiv.org/abs/1902.00434
"""

import torch
import torch.nn as nn
import numpy as np


def linear_interpolation(x, y, xnew, eps=1e-12):
    """Linear interpolation for sorted sequences.

    Args:
        x: [BL, N] sorted x coordinates
        y: [BL, N] y values at x coordinates
        xnew: [BL, M] new x coordinates to interpolate
        eps: small value for numerical stability

    Returns:
        ynew: [BL, M] interpolated y values
    """
    BL, N = x.shape

    if not x.is_contiguous():
        x = x.contiguous()
    if not xnew.is_contiguous():
        xnew = xnew.contiguous()

    ind = torch.searchsorted(x, xnew)
    ind = torch.clamp(ind, 1, N - 1)
    iL = ind - 1
    iR = ind

    x_left = torch.gather(x, 1, iL)
    x_right = torch.gather(x, 1, iR)
    y_left = torch.gather(y, 1, iL)
    y_right = torch.gather(y, 1, iR)

    slopes = (y_right - y_left) / ((x_right - x_left) + eps)
    ynew = y_left + slopes * (xnew - x_left)
    return ynew


class SWE_Pooling(nn.Module):
    """Sliced-Wasserstein Embedding pooling layer.

    Produces fixed-dimensional permutation-invariant embeddings for input sets
    of arbitrary size based on sliced-Wasserstein distance.

    Args:
        d_in: Dimensionality of input features (e.g., 11 for USV call features)
        num_slices: Number of random projection directions (L)
        num_ref_points: Number of points in the reference distribution (M)
        freeze_swe: If True, projection directions and reference are fixed
        flatten: If True, output is L*M dimensional; else L dimensional
    """

    def __init__(self, d_in, num_slices, num_ref_points, freeze_swe=False, flatten=True):
        super(SWE_Pooling, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.L = num_slices
        self.flatten = flatten

        # Reference points evenly distributed between -1 and 1
        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, num_slices)
        self.reference = nn.Parameter(uniform_ref)

        # Projection directions (unit vectors via weight normalization)
        self.theta = nn.utils.parametrizations.weight_norm(
            nn.Linear(d_in, num_slices, bias=False), dim=0
        )

        # Fix magnitude to 1 (unit vectors)
        self.theta.parametrizations.weight.original0.data = torch.ones_like(
            self.theta.parametrizations.weight.original0.data, requires_grad=False
        )
        self.theta.parametrizations.weight.original0.requires_grad = False

        # Initialize directions with Gaussian
        nn.init.normal_(self.theta.parametrizations.weight.original1)

        if freeze_swe:
            self.theta.parametrizations.weight.original1.requires_grad = False
            self.reference.requires_grad = False

        if not flatten:
            self.weight = nn.Linear(num_ref_points, 1, bias=False)

    @property
    def output_dim(self):
        """Return output dimension."""
        if self.flatten:
            return self.L * self.num_ref_points
        else:
            return self.L

    def get_slice(self, X):
        """Project samples onto slice directions."""
        return self.theta(X)

    def forward(self, X, mask=None, eps_val=1e-3):
        """
        Args:
            X: [B, N, d_in] batch of sets (B sets, each up to N samples)
            mask: [B, N] boolean mask, True for valid samples

        Returns:
            embeddings: [B, L*M] or [B, L] pooled embeddings
        """
        B, N, _ = X.shape
        M, _ = self.reference.shape

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=X.device)

        valid_counts = mask.sum(dim=1)
        zero_mask = (valid_counts == 0)

        # Handle empty sets
        if zero_mask.all():
            if self.flatten:
                return torch.zeros(B, self.L * M, device=X.device)
            else:
                return torch.zeros(B, self.L, device=X.device)

        single_mask = (valid_counts == 1)
        multi_mask = (valid_counts > 1)

        # Handle single-element sets
        if N == 1 and single_mask.any():
            X = torch.cat([X, X.clone()], dim=1)
            mask = torch.cat([mask, ~mask.clone()], dim=1)
            _, N, _ = X.shape

        Xslices = self.get_slice(X)
        Xslices_modified = Xslices.clone()
        invalid_projections = ~mask.unsqueeze(-1).expand(-1, -1, self.L)

        # Single valid: replicate
        if single_mask.any():
            valid_slices = Xslices_modified[single_mask, 0]
            valid_slices_expanded = valid_slices.unsqueeze(1).expand(-1, N, -1)
            Xslices_modified[single_mask] = valid_slices_expanded

        # Multi valid: slope logic for invalid entries
        if multi_mask.any():
            Xslices_sub = Xslices_modified[multi_mask].clone()
            inv_mask_sub = invalid_projections[multi_mask]

            Xslices_sub[inv_mask_sub] = -1e10
            top2, _ = torch.topk(Xslices_sub, k=2, dim=1)
            max_Xslices = top2[:, 0].unsqueeze(1)
            delta_y = -(top2[:, 1] - top2[:, 0]).unsqueeze(1)

            Xslices_sub = Xslices[multi_mask].clone()
            Xslices_sub[inv_mask_sub] = max_Xslices.expand(-1, N, -1)[inv_mask_sub]

            valid_counts_multi = valid_counts[multi_mask]
            delta_x = 1.0 / (1.0 + valid_counts_multi.unsqueeze(1))
            slope = delta_y / delta_x.unsqueeze(-1)
            slope = slope.expand(-1, N, -1)

            x_shifts_sub = eps_val * torch.cumsum(inv_mask_sub, dim=1)
            y_shifts = slope * x_shifts_sub
            Xslices_sub = Xslices_sub + y_shifts

            Xslices_modified[multi_mask] = Xslices_sub

        # Sort slices
        Xslices_sorted, _ = torch.sort(Xslices_modified, dim=1)

        # Normalize x coordinates
        x = torch.arange(1, N + 1, device=X.device).float()
        x = x.unsqueeze(0).expand(B, N)
        denom = 1.0 + valid_counts.unsqueeze(1)
        x = x / denom

        x_copy = x.clone()
        x_copy[~mask] = -1e10
        max_x, _ = torch.max(x_copy, dim=1, keepdim=True)
        x[~mask] = max_x.expand(-1, N)[~mask]

        x_shifts = eps_val * torch.cumsum(invalid_projections, dim=1)
        x = x.unsqueeze(2).expand(B, N, self.L).transpose(1, 2)
        x = x + x_shifts.transpose(1, 2)
        x = x.reshape(B * self.L, N)

        xnew = torch.linspace(0, 1, M + 2, device=X.device)[1:-1]
        xnew = xnew.unsqueeze(0).expand(B * self.L, -1)
        y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.L, -1).requires_grad_()

        # Interpolate
        ynew = linear_interpolation(x, y, xnew)

        Xslices_sorted_interpolated = torch.transpose(
            ynew.view(B, self.L, -1), 1, 2
        )

        # Compute Monge couplings
        Rslices = self.reference.expand(B, M, self.L)
        Rsorted, _ = torch.sort(Rslices, dim=1)
        monge_couplings = (Rsorted - Xslices_sorted_interpolated).transpose(1, 2)

        if self.flatten:
            embeddings = monge_couplings.reshape(B, -1)
        else:
            embeddings = self.weight(monge_couplings).squeeze(-1)

        if zero_mask.any():
            embeddings[zero_mask] = 0

        return embeddings


class SWEPooler:
    """Wrapper for SWE_Pooling that matches the CallPooler interface."""

    def __init__(self, n_features=11, num_slices=8, num_ref_points=10,
                 freeze_swe=False, flatten=True, device='cpu'):
        self.n_features = n_features
        self.num_slices = num_slices
        self.num_ref_points = num_ref_points
        self.flatten = flatten
        self.device = device

        self.swe = SWE_Pooling(
            d_in=n_features,
            num_slices=num_slices,
            num_ref_points=num_ref_points,
            freeze_swe=freeze_swe,
            flatten=flatten
        ).to(device)

        # Set to eval mode for non-learnable pooling
        if freeze_swe:
            self.swe.eval()

    @property
    def output_dim(self):
        return self.swe.output_dim

    def pool(self, call_features: np.ndarray) -> np.ndarray:
        """Pool call features using SWE.

        Args:
            call_features: (n_calls, n_features) numpy array

        Returns:
            (output_dim,) pooled features
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            X = torch.tensor(call_features, dtype=torch.float32, device=self.device)
            X = X.unsqueeze(0)  # (1, n_calls, n_features)

            # Create mask (all valid)
            mask = torch.ones(1, X.shape[1], dtype=torch.bool, device=self.device)

            # Pool
            pooled = self.swe(X, mask)  # (1, output_dim)

            return pooled.squeeze(0).cpu().numpy()

    def __repr__(self):
        return f"SWEPooler(output_dim={self.output_dim}, slices={self.num_slices}, refs={self.num_ref_points})"
