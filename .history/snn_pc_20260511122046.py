"""
snn_pc.py — Spiking Neural Network for Predictive Coding
Faithful replication of Lee et al. (2024)
Frontiers in Computational Neuroscience 18:1338280

Architecture
    Area 0 (784)  →  Area 1 (400)  →  Area 2 (225)  →  Area 3 (64)
    Each area l>0: R^l (representation), E+^l, E-^l (error units)
    FFG pathway:   R^0 → G (16 gist units) → R^l  [fixed, sparse]   (currently disabled)
    Learning:      Hebbian on inter-areal weights W^{l,l+1}
    Neurons:       AdEx (adaptive exponential integrate-and-fire)
    Synapses:      AMPA + NMDA kinetics (rise 5 ms, decay 50 ms)

This file is the BATCHED rewrite. Every neuron group carries a leading
batch dimension B; the whole T-step rollout for B samples is one Python
loop of length T (not B*T), and every Hebbian update is a single matmul
per layer instead of B outer products.

Single-sample callers (predict_class, run_sample_full) are thin wrappers
over the batched core with B=1.

FFG is not yet implemented in the batched path; constructing with
use_ffg=True raises NotImplementedError.

FIX vs original:
    PCArea weight init std changed from 0.3/n_R_above → 1.0/n_R_above.
    The original value produced ~280 pA to Area 1 (below the ~600 pA
    spike threshold), silently preventing all signal propagation beyond
    Area 0. 1.0/n_R_above delivers ~900 pA, comfortably above threshold.

This file has been adapted to run the core dynamics and learning in
PyTorch so the model can take advantage of GPU acceleration where
available. The public API and entry points remain numpy-friendly via
small conversion boundaries at the edges.
"""

from __future__ import annotations

import os
import time
import warnings

import matplotlib
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# AdEx parameters (Table 1, Brette & Gerstner 2005)
# ─────────────────────────────────────────────────────────────
CM = 281e-12
GL = 30e-9
EL = -70.6e-3
VT = -50.4e-3
DELTA_T = 2e-3
TREF = 2
C_ADP = 4e-9
B_ADP = 0.0805e-9
TAU_A = 144e-3
TAU_RISE = 5e-3
TAU_DEC = 50e-3
VR = -70.6e-3

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor(value, *, device=None, dtype=None):
    if device is None:
        device = DEFAULT_DEVICE
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────
# 1.  AdEx Neuron Group  (batched: state shape (B, n))
# ─────────────────────────────────────────────────────────────


class NeuronGroup:
    def __init__(self, n: int, dt: float = 1e-3, B: int = 1, device=None):
        self.n = n
        self.B = B
        self.dt = dt
        self.device = device if device is not None else DEFAULT_DEVICE
        self._alloc(B)

    def _alloc(self, B: int):
        self.B = B
        self.V = torch.full((B, self.n), EL, dtype=torch.float32, device=self.device)
        self.a = torch.zeros((B, self.n), dtype=torch.float32, device=self.device)
        self.Y = torch.zeros((B, self.n), dtype=torch.float32, device=self.device)
        self.X = torch.zeros((B, self.n), dtype=torch.float32, device=self.device)
        self.spk = torch.zeros((B, self.n), dtype=torch.bool, device=self.device)
        self.ref = torch.zeros((B, self.n), dtype=torch.int32, device=self.device)

    def step(self, I_ext):
        I_ext = _to_tensor(I_ext, device=self.device, dtype=torch.float32)
        dt = self.dt
        V = self.V
        a = self.a
        exp_arg = torch.clamp((V - VT) / DELTA_T, -30.0, 20.0)
        exp_term = GL * DELTA_T * torch.exp(exp_arg)
        dV = (-GL * (V - EL) + exp_term + I_ext - a) / CM * dt
        da = (C_ADP * (V - EL) - a) / TAU_A * dt
        self.V.add_(dV)
        self.a.add_(da)

        in_ref = self.ref > 0
        self.V[in_ref] = VR
        self.ref[in_ref] -= 1
        fired = (self.V > VT) & (~in_ref)
        self.spk.copy_(fired)
        self.V[fired] = VR
        self.a[fired] += B_ADP
        self.ref[fired] = TREF
        self.Y[fired] = 1.0

        dX = (self.Y / TAU_RISE - self.X / TAU_DEC) * dt
        dY = (-self.Y / TAU_DEC) * dt
        self.X.add_(dX)
        self.Y.add_(dY)
        self.X.clamp_(min=0.0)
        self.Y.clamp_(min=0.0)

    def reset(self, B: int | None = None):
        if B is not None and B != self.B:
            self._alloc(B)
            return
        self.V.fill_(EL)
        self.a.zero_()
        self.Y.zero_()
        self.X.zero_()
        self.spk.zero_()
        self.ref.zero_()


class FFGPathway:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FFG pathway is not implemented in the torch backend. "
            "Run with use_ffg=False (CLI: --no-ffg)."
        )


class PCArea:
    def __init__(self, l: int, n_R: int, n_R_above=None, B: int = 1, rng=None, device=None):
        if rng is None:
            rng = np.random.default_rng()
        self.l = l
        self.n_R = n_R
        self.device = device if device is not None else DEFAULT_DEVICE
        self.R = NeuronGroup(n_R, B=B, device=self.device)
        self.EP = NeuronGroup(n_R, B=B, device=self.device)
        self.EN = NeuronGroup(n_R, B=B, device=self.device)
        if n_R_above is not None:
            scaled_std = 1.0 / n_R_above
            raw = rng.normal(0.0, scaled_std, (n_R, n_R_above)).astype(np.float32)
            self.W = torch.as_tensor(np.abs(raw), device=self.device)
        else:
            self.W = None

    def reset(self, B: int | None = None):
        self.R.reset(B)
        self.EP.reset(B)
        self.EN.reset(B)


class SNNPC:
    def __init__(self, area_sizes=None, n_gist=16,
                 dt=1e-3, lr=1e-7, reg=1e-5,
                 tw_ms=100, T_ms=350,
                 use_ffg=True, syn_gain=1000.0,
                 use_class_area=False, n_classes=10,
                 cls_clamp_gain=800.0, rng=None,
                 device=None):
        if area_sizes is None:
            area_sizes = [784, 400, 225, 64]
        if use_class_area:
            area_sizes.append(n_classes)
        if rng is None:
            rng = np.random.default_rng()
        if use_ffg:
            raise NotImplementedError(
                "use_ffg=True is not supported in the torch backend. "
                "Pass use_ffg=False (CLI: --no-ffg)."
            )
        self.device = device if device is not None else DEFAULT_DEVICE
        self.area_sizes = area_sizes
        self.L = len(area_sizes)
        self.dt = dt
        self.lr = lr
        self.reg = reg
        self.tw = int(tw_ms / (dt * 1000))
        self.T = int(T_ms / (dt * 1000))
        self.use_ffg = False
        self.I_scale = 1e-12
        self.syn_gain = syn_gain
        self.use_class_area = use_class_area
        self.n_classes = n_classes
        self.cls_clamp_gain = cls_clamp_gain
        self.areas = []
        for l in range(self.L):
            n_above = area_sizes[l + 1] if l < self.L - 1 else None
            self.areas.append(PCArea(l, area_sizes[l], n_above, B=1, rng=rng, device=self.device))

    def label_clamp_current(self, labels, gain: float | None = None):
        if not self.use_class_area:
            raise ValueError("Classification area is disabled.")
        if gain is None:
            gain = self.cls_clamp_gain
        labels = _to_tensor(labels, device=self.device, dtype=torch.long).reshape(-1)
        B = labels.shape[0]
        n_cls = self.area_sizes[-1]
        clamp = torch.zeros((B, n_cls), dtype=torch.float32, device=self.device)
        clamp[torch.arange(B, device=self.device), labels] = float(gain)
        return clamp

    def reset_all(self, B: int):
        for area in self.areas:
            area.reset(B)

    def step(self, pixel_pA, class_label=None, class_clamp_gain: float | None = None):
        areas = self.areas
        gain = self.syn_gain
        scale = self.I_scale

        pixel_pA = _to_tensor(pixel_pA, device=self.device, dtype=torch.float32)
        areas[0].R.step(pixel_pA * scale)

        for l in range(self.L - 1):
            W = areas[l].W
            R_above = areas[l + 1].R.X
            R_below = areas[l].R.X

            pred = R_above @ W.T
            areas[l].EP.step((R_below - pred) * gain * scale)
            areas[l].EN.step((pred - R_below) * gain * scale)

            bu_p = areas[l].EP.X @ W
            bu_n = areas[l].EN.X @ W

            if l + 1 < self.L - 1:
                td_p = areas[l + 1].EP.X
                td_n = areas[l + 1].EN.X
            else:
                td_p = torch.zeros_like(bu_p)
                td_n = torch.zeros_like(bu_n)

            I_R = (bu_p - bu_n - td_p + td_n) * gain * scale

            if (self.use_class_area and class_label is not None and (l + 1) == (self.L - 1)):
                I_R = self.label_clamp_current(class_label, class_clamp_gain) * scale

            areas[l + 1].R.step(I_R)

    def run_batch(self, pixel_pA, class_label=None, class_clamp_gain: float | None = None):
        pixel_pA = _to_tensor(pixel_pA, device=self.device, dtype=torch.float32)
        if pixel_pA.ndim != 2:
            raise ValueError(f"pixel_pA must be (B, n0); got shape {tuple(pixel_pA.shape)}")
        B = pixel_pA.shape[0]

        if class_label is not None:
            class_label = _to_tensor(class_label, device=self.device, dtype=torch.long).reshape(-1)
            if class_label.shape[0] != B:
                raise ValueError(
                    f"class_label batch dim {class_label.shape[0]} != pixel_pA batch dim {B}"
                )

        self.reset_all(B)
        R_buf = {l: [] for l in range(self.L)}
        EP_buf = {l: [] for l in range(self.L - 1)}
        EN_buf = {l: [] for l in range(self.L - 1)}

        for t in range(self.T):
            self.step(pixel_pA, class_label=class_label, class_clamp_gain=class_clamp_gain)
            if t >= self.T - self.tw:
                for l in range(self.L):
                    R_buf[l].append(self.areas[l].R.X.detach().clone())
                for l in range(self.L - 1):
                    EP_buf[l].append(self.areas[l].EP.X.detach().clone())
                    EN_buf[l].append(self.areas[l].EN.X.detach().clone())

        X_R = {l: torch.stack(R_buf[l], dim=0).mean(dim=0) for l in range(self.L)}
        X_EP = {l: torch.stack(EP_buf[l], dim=0).mean(dim=0) for l in range(self.L - 1)}
        X_EN = {l: torch.stack(EN_buf[l], dim=0).mean(dim=0) for l in range(self.L - 1)}
        return X_R, X_EP, X_EN

    def run_sample_full(self, pixel_pA, class_label: int | None = None, class_clamp_gain: float | None = None):
        pA_B = _to_tensor(pixel_pA, device=self.device, dtype=torch.float32).reshape(1, -1)
        lbl_B = torch.tensor([int(class_label)], device=self.device, dtype=torch.long) if class_label is not None else None
        X_R, X_EP, X_EN = self.run_batch(pA_B, lbl_B, class_clamp_gain)
        X_R = {l: _to_numpy(X_R[l][0]) for l in range(self.L)}
        X_EP = {l: _to_numpy(X_EP[l][0]) for l in range(self.L - 1)}
        X_EN = {l: _to_numpy(X_EN[l][0]) for l in range(self.L - 1)}
        return X_R, X_EP, X_EN

    def predict_classes(self, pixel_pA):
        if not self.use_class_area:
            raise ValueError("Classification area is disabled.")
        X_R, _, _ = self.run_batch(pixel_pA, class_label=None)
        return _to_numpy(torch.argmax(X_R[self.L - 1], dim=1)).astype(int)

    def predict_class(self, pixel_pA):
        return int(self.predict_classes(np.asarray(pixel_pA)[None, :])[0])

    def weight_update(self, X_R: dict, X_EP: dict, X_EN: dict):
        B = next(iter(X_R.values())).shape[0]
        for l in range(self.L - 1):
            ep = _to_tensor(X_EP[l], device=self.device, dtype=torch.float32)
            en = _to_tensor(X_EN[l], device=self.device, dtype=torch.float32)
            r = _to_tensor(X_R[l + 1], device=self.device, dtype=torch.float32)
            dW_p = ep.T @ r / B
            dW_n = en.T @ r / B
            g_W = (self.areas[l].W > 0).to(dtype=self.areas[l].W.dtype)
            dW = self.lr * (dW_p - dW_n) - self.reg * g_W
            self.areas[l].W.add_(dW)
            self.areas[l].W.clamp_(min=0.0)


def preprocess_image(img_flat, lo: float = 600.0, hi: float = 3000.0):
    x = _to_tensor(img_flat, device="cpu", dtype=torch.float32) / 255.0
    n = torch.linalg.norm(x)
    if float(n) > 1e-10:
        x = x / n
    xmin = torch.min(x)
    xmax = torch.max(x)
    if float(xmax - xmin) > 0.0:
        x = (x - xmin) / (xmax - xmin)
    return _to_numpy(x * (hi - lo) + lo)


def preprocess_batch(X):
    return np.stack([preprocess_image(X[i]) for i in range(len(X))])


def compute_nrmse(actual, predicted):
    actual_t = _to_tensor(actual, device="cpu", dtype=torch.float32)
    predicted_t = _to_tensor(predicted, device="cpu", dtype=torch.float32)
    rmse = torch.sqrt(torch.mean((actual_t - predicted_t) ** 2, dim=-1))
    r = torch.max(actual_t, dim=-1).values - torch.min(actual_t, dim=-1).values
    if rmse.ndim == 0:
        return torch.tensor(0.0) if float(r) < 1e-10 else rmse / r
    out = torch.zeros_like(rmse)
    mask = r > 1e-10
    out[mask] = rmse[mask] / r[mask]
    return out


def train_snn_pc(model: SNNPC, X_train, y_train,
                 n_epochs=50, batch_size=32,
                 verbose=True, log_interval=5,
                 use_classification=False,
                 class_clamp_gain: float | None = None,
                 rng=None):
    N = len(X_train)
    n_batches = N // batch_size
    if rng is None:
        rng = np.random.default_rng()
    history = {
        "nrmse": {l: [] for l in range(model.L - 1)},
        "cls_acc": [],
        "epoch_time": [],
    }

    for epoch in range(n_epochs):
        t0 = time.time()
        perm = rng.permutation(N)
        ep_nrmse = {l: [] for l in range(model.L - 1)}
        ep_cls_acc = []

        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            pA_B = preprocess_batch(X_train[idx])

            label_B = None
            if use_classification and model.use_class_area:
                label_B = y_train[idx].astype(np.int64)

            X_R, X_EP, X_EN = model.run_batch(
                pA_B,
                class_label=label_B,
                class_clamp_gain=class_clamp_gain,
            )

            if use_classification and model.use_class_area:
                preds = torch.argmax(X_R[model.L - 1], dim=1)
                ep_cls_acc.extend((_to_numpy(preds == _to_tensor(label_B, device=model.device, dtype=torch.long))).astype(float).tolist())

            for l in range(model.L - 1):
                pred_l = X_R[l + 1] @ model.areas[l].W.T
                ep_nrmse[l].extend(_to_numpy(compute_nrmse(X_R[l], pred_l)).tolist())

            model.weight_update(X_R, X_EP, X_EN)

        elapsed = time.time() - t0
        for l in range(model.L - 1):
            history["nrmse"][l].append(float(np.mean(ep_nrmse[l])))
        history["cls_acc"].append(
            float(np.mean(ep_cls_acc))
            if (use_classification and model.use_class_area and len(ep_cls_acc) > 0)
            else float("nan")
        )
        history["epoch_time"].append(elapsed)

        if verbose and (epoch % log_interval == 0 or epoch == n_epochs - 1):
            nstr = "  |  ".join(
                f"Area{l+1}: {history['nrmse'][l][-1]:.4f}"
                for l in range(model.L - 1)
            )
            if use_classification and model.use_class_area:
                print(
                    f"  Epoch {epoch+1:3d}/{n_epochs}  |  {nstr}"
                    f"  |  ClsAcc: {history['cls_acc'][-1]:.3f}  |  {elapsed:.1f}s"
                )
            else:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}  |  {nstr}  |  {elapsed:.1f}s")
    return history


def compute_rdm(representations):
    reps = _to_numpy(representations)
    Xr = rankdata(reps, axis=1, method="average")
    X = Xr - Xr.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-10)
    X = X / nrm
    sim = X @ X.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def second_order_rsa(rdm_a, rdm_b) -> float:
    rdm_a = _to_numpy(rdm_a)
    rdm_b = _to_numpy(rdm_b)
    N = rdm_a.shape[0]
    idx = np.triu_indices(N, k=1)
    rho, _ = spearmanr(rdm_a[idx], rdm_b[idx])
    return float(rho)


def add_noise(X, sigma_pA: float = 300.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sigma_px = sigma_pA * 255.0 / 3000.0
    noise = rng.normal(0.0, sigma_px, np.asarray(X).shape)
    return np.clip(np.asarray(X, dtype=np.float32) + noise, 0.0, 255.0)


def add_occlusion(X, patch_size: int = 9, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    out = np.asarray(X, dtype=np.float32).copy()
    imgs = out.reshape(-1, 28, 28)
    for img in imgs:
        r = rng.integers(0, 28 - patch_size)
        c = rng.integers(0, 28 - patch_size)
        img[r:r + patch_size, c:c + patch_size] = 0.0
    return imgs.reshape(out.shape)


def get_representations(model: SNNPC, X,
                        area: int = 1, verbose: bool = False,
                        chunk_size: int = 32):
    N = len(X)
    R_list = [None] * N
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pA_B = preprocess_batch(X[start:end])
        X_R, _, _ = model.run_batch(pA_B)
        for j in range(end - start):
            R_list[start + j] = {l: _to_numpy(X_R[l][j]) for l in range(model.L)}
        if verbose:
            print(f"    {end}/{N}")
    R_mat = np.array([R_list[i][area] for i in range(N)])
    return R_mat, R_list


def linear_decode(R_tr, y_tr, R_te, y_te, random_state=None):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    clf.fit(_to_numpy(R_tr), _to_numpy(y_tr))
    return float(clf.score(_to_numpy(R_te), _to_numpy(y_te))), clf
