# gmm_module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gmm_fitting(dino_features, y_hat, temp=100.0, return_adapter=False, max_iter=10):
    K = y_hat.size(1)
    num_samples, d = dino_features.size()
    device = dino_features.device

    std_init = 1.0 / (d ** 0.5)
    lambda_value = 1.0

    y_hat = y_hat * 70
    y_hat, z = init_z(y_hat, softmax=True)

    mu = init_mu(K, d, z, dino_features)
    std = init_sigma(d, std_init, device)
    adapter = Gaussian(mu=mu, std=std).to(device)

    for k in range(max_iter + 1):
        gmm_likelihood = adapter(dino_features, no_exp=True)
        z, loss = update_z_wo_graph(gmm_likelihood, y_hat, z, lambda_value, temp)[0:num_samples]
        if k == max_iter:
            break
        adapter = update_mu(adapter, dino_features, z)
        adapter = update_sigma(adapter, dino_features, z)

    if return_adapter:
        return z, adapter
    return z


def update_z_wo_graph(gmm_likelihood, y_hat, z, lambda_value, temp=50.0):
    num_samples = gmm_likelihood.size(0)
    intermediate = gmm_likelihood.clone()
    intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]

    q = torch.exp(1 / 50 * intermediate)
    q = q / torch.sum(q, dim=1, keepdim=True)

    intermediate = y_hat * torch.exp(1 / temp * intermediate)
    z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)

    p = y_hat
    kl_zp = F.kl_div(input=p.log(), target=z, reduction='batchmean')
    kl_zq = F.kl_div(input=q.log(), target=z, reduction='batchmean')
    loss = kl_zp + kl_zq
    return z, loss


class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.mu = mu.clone()
        self.K, self.num_components, self.d = self.mu.shape
        self.std = std.clone()
        self.mixing = torch.ones(self.K, self.num_components, device=self.mu.device) / self.num_components

    def forward(self, x, get_components=False, no_exp=False):
        chunk_size = 2500
        N = x.shape[0]
        M = self.mu.shape[0]
        intermediate = torch.empty((N, M), dtype=x.dtype, device=x.device)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            intermediate[s:e] = -0.5 * torch.einsum(
                'ijk,ijk->ij',
                (x[s:e][:, None, :] - self.mu[None, :, 0, :]) ** 2,
                1 / self.std[None, None, :])
        if not no_exp:
            intermediate = torch.exp(intermediate)
        if get_components:
            return torch.ones_like(intermediate.unsqueeze(1))
        return intermediate

    def set_std(self, std):
        self.std = std


def update_mu(adapter, query_features, z):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    weights = (1 / n_query) * affinity_unlabeled
    new_mu = torch.einsum('ij,ik->jk', weights, query_features)
    new_mu /= (1 / n_query * torch.sum(affinity_unlabeled, dim=0).unsqueeze(-1))
    new_mu = new_mu.unsqueeze(1)
    new_mu /= new_mu.norm(dim=-1, keepdim=True)
    adapter.mu = new_mu
    return adapter


def update_sigma(adapter, query_features, z):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    std = 0
    chunk_size = 2500
    for s in range(0, n_query, chunk_size):
        e = min(s + chunk_size, n_query)
        q_chunk = query_features[s:e]
        chunk_result = (1 / n_query) * torch.einsum(
            'ij,ijk->k', affinity_unlabeled[s:e, :],
            (q_chunk[:, None, :] - adapter.mu[None, :, 0, :]) ** 2)
        if s == 0:
            std = chunk_result
        else:
            std += chunk_result
    std /= (1 / n_query * torch.sum(affinity_unlabeled[:, :]))
    adapter.set_std(std)
    return adapter


def init_z(affinity, softmax=True):
    y_hat = F.softmax(affinity, dim=1) if softmax else affinity
    z = y_hat.clone()
    return y_hat, z


def init_mu(K, d, z, query_features):
    mu = torch.zeros(K, 1, d, device=query_features.device)
    n_most_confident = 8
    topk_values, topk_indices = torch.topk(z, k=n_most_confident, dim=0)
    mask = torch.zeros_like(z).scatter_(0, topk_indices, 1)
    filtered_z = z * mask
    for c in range(K):
        class_indices = mask[:, c].nonzero().squeeze(1)
        class_features = query_features[class_indices]
        class_z = filtered_z[class_indices, c].unsqueeze(1)
        combined = class_features * class_z
        component_mean = combined[:n_most_confident].mean(dim=0)
        mu[c, 0, :] = component_mean
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def init_sigma(d, std_init, device):
    return (torch.eye(d, device=device).diag() * std_init)