import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
import scipy.linalg
from datasetLoaders import make_cifar10_eval_loader

class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        net = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        net.fc = torch.nn.Identity()  # output 2048-d features
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(device)

        # Inception normalization constants (ImageNet)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @torch.no_grad()
    def forward(self, x_bchw):
        # x expected in [-1,1] (your CIFAR transform does that)
        x = (x_bchw + 1) / 2.0                      # -> [0,1]
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.net(x)                         # (B, 2048)
        return feats

def compute_stats_from_features(features_np: np.ndarray):
    # features_np shape: (N, D)
    mu = features_np.mean(axis=0)
    diff = features_np - mu
    sigma = diff.T @ diff / (features_np.shape[0] - 1)
    return mu, sigma



def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # sqrtm can be numerically unstable; add eps to diagonals
    covmean, _ = scipy.linalg.sqrtm((sigma1 + eps*np.eye(sigma1.shape[0])) @
                                   (sigma2 + eps*np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)
    return float(fid)

def compute_real_stats(device, n_real=50000, batch_size=128):
    loader = make_cifar10_eval_loader(batch_size=batch_size)
    feat_net = InceptionFeatureExtractor(device=device)

    feats = []
    seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        f = feat_net(x).cpu().numpy()
        feats.append(f)
        seen += f.shape[0]
        if seen >= n_real:
            break

    feats_np = np.concatenate(feats, axis=0)[:n_real]
    mu, sigma = compute_stats_from_features(feats_np)
    return mu, sigma

def make_fid_fn(device, sample_fn, real_mu, real_sigma,
                n_gen=5000, gen_batch=50):  # 5k is cheap; 50k is paper-accurate but expensive
    feat_net = InceptionFeatureExtractor(device=device)

    def fid_fn(model):
        feats = []
        remaining = n_gen
        while remaining > 0:
            b = min(gen_batch, remaining)
            x = sample_fn(model, B=b, C=3, H=32, W=32, device=device)  # returns cpu in your code
            x = x.to(device)  # move back to GPU for inception
            f = feat_net(x).cpu().numpy()
            feats.append(f)
            remaining -= b

        feats_np = np.concatenate(feats, axis=0)
        mu_g, sigma_g = compute_stats_from_features(feats_np)
        return frechet_distance(real_mu, real_sigma, mu_g, sigma_g)

    return fid_fn

