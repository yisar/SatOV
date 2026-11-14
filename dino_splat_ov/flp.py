"""
FLP: Feature-space Label Propagation (Prior-Guided)

Replaces the GMM-based window refinement. On a sparse kNN graph over DINO patch
features, we propagate the text-logit prior via the classical label-propagation
closed form (Zhou et al. 2004):

    F* = argmin_F  ||F - Y||_F^2  +  λ · tr(F^T L_w F)
       = (I + λ L_w)^{-1} Y
       ≈ (1-α)(I - α W_n)^{-1} Y           with α = 1/(1+λ)

We solve it by the standard power iteration F ← α W_n F + (1-α) Y, which is
memory-free and stops in `iters` steps. Optionally, the feature affinity W is
gated by the semantic similarity between patches' text logits, so that
cross-class diffusion is suppressed — this is the "prior-guided" part.

Interface is drop-in compatible with the old gmm_fitting:
    z = feature_label_propagation(dino_features, y_hat, ...)
    # dino_features: [N, D]   raw patch features (unnormalized ok)
    # y_hat:         [N, C]   raw text logits
    # z:             [N, C]   refined soft assignment, rows sum to 1
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------
def _build_knn_affinity(
    feats: torch.Tensor, k: int, sigma_f: float
) -> torch.Tensor:
    """
    Sparse symmetric kNN affinity matrix over cosine similarity.

    W_ij = exp( -(1 - cos(f_i, f_j)) / sigma_f )    if j in kNN(i) or i in kNN(j)
         = 0                                        otherwise
    """
    N = feats.size(0)
    sim = feats @ feats.t()                                    # [N,N]
    k_eff = min(k + 1, N)                                      # +1 for self

    topk_v, topk_i = sim.topk(k_eff, dim=-1)                   # [N,k_eff]
    w = torch.exp(-(1.0 - topk_v) / max(sigma_f, 1e-6))        # in (0,1]

    W = torch.zeros_like(sim)
    W.scatter_(1, topk_i, w)
    W = (W + W.t()) * 0.5                                      # symmetrize
    return W


def _semantic_gate(y_hat: torch.Tensor, sigma_y: float) -> torch.Tensor:
    """
    Pairwise semantic-similarity gate from the text prior.
    Gate_ij = exp( -(1 - cos(y_i, y_j)) / sigma_y )  ∈ (0, 1]
    """
    y = F.normalize(y_hat, dim=-1)
    sim = y @ y.t()                                            # [N,N] in [-1,1]
    return torch.exp(-(1.0 - sim) / max(sigma_y, 1e-6))


def _sym_normalize(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    d = W.sum(dim=-1).clamp_min(eps)
    d_inv_sqrt = d.rsqrt()
    return d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def feature_label_propagation(
    dino_features: torch.Tensor,
    y_hat: torch.Tensor,
    k: int = 8,
    alpha: float = 0.9,
    iters: int = 20,
    sigma_f: float = 0.1,
    sigma_y: float = 0.2,
    prior_scale: float = 20.0,
    use_semantic_affinity: bool = True,
    return_affinity: bool = False,
) -> torch.Tensor:
    """
    Prior-guided label propagation on a sparse feature-space kNN graph.

    Args:
        dino_features: [N, D]  DINO patch features
        y_hat:         [N, C]  text logits (prior / data term)
        k:             kNN size
        alpha:         propagation strength ∈ (0,1); 0 = pure prior, 1 = pure diffusion
        iters:         power-iteration steps (closed form is the limit iter→∞)
        sigma_f:       feature-graph bandwidth on (1 - cos_sim)
        sigma_y:       semantic-gate bandwidth on (1 - cos_sim of text logits)
        prior_scale:   temperature applied to y_hat before softmax (sharpens prior)
        use_semantic_affinity: gate W_ij by text-logit similarity
        return_affinity:       also return the normalized affinity W_n

    Returns:
        z:  [N, C]  refined soft assignment
        (and W_n if return_affinity=True)
    """
    device = dino_features.device
    N = dino_features.size(0)

    # 1) sparse feature graph
    feats = F.normalize(dino_features.float(), dim=-1)
    W = _build_knn_affinity(feats, k=k, sigma_f=sigma_f)

    # 2) semantic gating from the text prior
    if use_semantic_affinity:
        W = W * _semantic_gate(y_hat.float(), sigma_y=sigma_y)

    # 3) symmetric normalization -> W_n
    Wn = _sym_normalize(W).to(dtype=dino_features.dtype)

    # 4) soft prior as data term
    Y = F.softmax(y_hat.float() * prior_scale, dim=-1).to(dtype=Wn.dtype)

    # 5) power iteration:  F ← α W_n F + (1-α) Y
    F_cur = Y
    for _ in range(max(1, int(iters))):
        F_cur = alpha * (Wn @ F_cur) + (1.0 - alpha) * Y

    z = F.softmax(F_cur, dim=-1)

    if return_affinity:
        return z, Wn
    return z