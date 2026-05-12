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
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
import os, time, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# AdEx parameters (Table 1, Brette & Gerstner 2005)
# ─────────────────────────────────────────────────────────────
CM       = 281e-12
GL       = 30e-9
EL       = -70.6e-3
VT       = -50.4e-3
DELTA_T  = 2e-3
TREF     = 2 # refractory period, each step is 1 ms -> 2 step refractory
C_ADP    = 4e-9
B_ADP    = 0.0805e-9
TAU_A    = 144e-3
TAU_RISE = 5e-3
TAU_DEC  = 50e-3
VR       = -70.6e-3


# ─────────────────────────────────────────────────────────────
# 1.  AdEx Neuron Group  (batched: state shape (B, n))
# ─────────────────────────────────────────────────────────────
class NeuronGroup:
    def __init__(self, n: int, dt: float = 1e-3, B: int = 1):
        self.n  = n
        self.B  = B
        self.dt = dt
        self._alloc(B)

    def _alloc(self, B: int):
        n = self.n
        self.B   = B
        self.V   = np.full((B, n), EL, dtype=np.float64)
        self.a   = np.zeros((B, n), dtype=np.float64)
        self.Y   = np.zeros((B, n), dtype=np.float64)
        self.X   = np.zeros((B, n), dtype=np.float64)
        self.spk = np.zeros((B, n), dtype=bool)
        self.ref = np.zeros((B, n), dtype=np.int32)

    def step(self, I_ext: np.ndarray):
        # I_ext: (B, n)
        dt = self.dt
        V  = self.V
        a  = self.a
        exp_arg  = np.clip((V - VT) / DELTA_T, -30.0, 20.0)
        exp_term = GL * DELTA_T * np.exp(exp_arg)
        dV = (-GL * (V - EL) + exp_term + I_ext - a) / CM * dt
        da = (C_ADP * (V - EL) - a) / TAU_A * dt
        self.V += dV
        self.a += da
        in_ref = self.ref > 0
        self.V[in_ref] = VR
        self.ref[in_ref] -= 1
        fired = (self.V > VT) & (~in_ref)
        self.spk[:] = fired
        self.V[fired]  = VR
        self.a[fired] += B_ADP
        self.ref[fired] = TREF
        self.Y[fired] = 1.0
        dX = (self.Y / TAU_RISE - self.X / TAU_DEC) * dt
        dY = (-self.Y / TAU_DEC) * dt
        self.X += dX
        self.Y += dY
        np.maximum(self.X, 0.0, out=self.X)
        np.maximum(self.Y, 0.0, out=self.Y)

    def reset(self, B: int | None = None):
        if B is not None and B != self.B:
            self._alloc(B)
            return
        self.V[:]   = EL
        self.a[:]   = 0.0
        self.Y[:]   = 0.0
        self.X[:]   = 0.0
        self.spk[:] = False
        self.ref[:] = 0


# ─────────────────────────────────────────────────────────────
# 2.  Feedforward Gist (FFG) Pathway  — NOT YET BATCHED
# ─────────────────────────────────────────────────────────────
class FFGPathway:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FFG pathway is not implemented in the batched code path. "
            "Run with use_ffg=False (CLI: --no-ffg)."
        )


# ─────────────────────────────────────────────────────────────
# 3.  Predictive-Coding Area
# ─────────────────────────────────────────────────────────────
class PCArea:
    def __init__(self, l: int, n_R: int, n_R_above=None, B: int = 1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.l   = l
        self.n_R = n_R
        self.R   = NeuronGroup(n_R, B=B)
        self.EP  = NeuronGroup(n_R, B=B)
        self.EN  = NeuronGroup(n_R, B=B)
        if n_R_above is not None:
            # FIX: std = 1.0/n_R_above  (original was 0.3/n_R_above, sub-threshold)
            scaled_std = 1.0 / n_R_above
            raw = rng.normal(0.0, scaled_std, (n_R, n_R_above))
            self.W = np.abs(raw)
        else:
            self.W = None

    def reset(self, B: int | None = None):
        self.R.reset(B)
        self.EP.reset(B)
        self.EN.reset(B)


# ─────────────────────────────────────────────────────────────
# 4.  Full SNN-PC Network  (batched)
# ─────────────────────────────────────────────────────────────
class SNNPC:
    def __init__(self, area_sizes=None, n_gist=16,
                 dt=1e-3, lr=1e-7, reg=1e-5,
                 tw_ms=100, T_ms=350,
                 use_ffg=True, syn_gain=1000.0,
                 use_class_area=False, n_classes=10,
                 cls_clamp_gain=800.0, rng=None):
        if area_sizes is None:
            area_sizes = [784, 400, 225, 64]
        if use_class_area:
            area_sizes.append(n_classes)
        if rng is None:
            rng = np.random.default_rng()
        if use_ffg:
            raise NotImplementedError(
                "use_ffg=True is not supported on the batched code path. "
                "Pass use_ffg=False (CLI: --no-ffg)."
            )
        self.area_sizes = area_sizes
        self.L          = len(area_sizes)
        self.dt         = dt
        self.lr         = lr
        self.reg        = reg
        self.tw         = int(tw_ms / (dt * 1000))
        self.T          = int(T_ms  / (dt * 1000))
        self.use_ffg    = False
        self.I_scale    = 1e-12
        self.syn_gain   = syn_gain
        self.use_class_area = use_class_area
        self.n_classes = n_classes
        self.cls_clamp_gain = cls_clamp_gain
        self.areas = []
        for l in range(self.L):
            n_above = area_sizes[l + 1] if l < self.L - 1 else None
            self.areas.append(PCArea(l, area_sizes[l], n_above, B=1, rng=rng))

    # ── label → class-area clamp current  (batched) ──────────
    def label_clamp_current(self, labels: np.ndarray,
                            gain: float | None = None) -> np.ndarray:
        """
        labels: (B,) array of int class indices.
        Returns (B, n_classes) clamp current in pA.
        """
        if not self.use_class_area:
            raise ValueError("Classification area is disabled.")
        if gain is None:
            gain = self.cls_clamp_gain
        labels = np.asarray(labels, dtype=np.int64)
        B = labels.shape[0]
        n_cls = self.area_sizes[-1]
        clamp = np.zeros((B, n_cls), dtype=np.float64)
        clamp[np.arange(B), labels] = gain
        return clamp

    # ── reset ────────────────────────────────────────────────
    def reset_all(self, B: int):
        for area in self.areas:
            area.reset(B)

    # ── one timestep over a batch of B samples ───────────────
    def step(self, pixel_pA: np.ndarray,
             class_label: np.ndarray | None = None,
             class_clamp_gain: float | None = None):

        areas = self.areas
        gain  = self.syn_gain
        scale = self.I_scale

        areas[0].R.step(pixel_pA * scale)

        for l in range(self.L - 1):
            W = areas[l].W                              # (n_l, n_l+1)
            R_above = areas[l + 1].R.X                  # (B, n_l+1)
            R_below = areas[l].R.X                      # (B, n_l)

            pred = R_above @ W.T                        # (B, n_l)
            areas[l].EP.step((R_below - pred) * gain * scale)
            areas[l].EN.step((pred - R_below) * gain * scale)

            bu_p = areas[l].EP.X @ W                    # (B, n_l+1)
            bu_n = areas[l].EN.X @ W

            if l + 1 < self.L - 1:
                td_p = areas[l + 1].EP.X                # (B, n_l+1)
                td_n = areas[l + 1].EN.X
            else:
                td_p = 0.0
                td_n = 0.0

            I_R = (bu_p - bu_n - td_p + td_n) * gain * scale

            if (self.use_class_area and class_label is not None
                    and (l + 1) == (self.L - 1)):
                # Overwrite with one-hot label clamp (matches set_I_R semantics).
                I_R = self.label_clamp_current(class_label, class_clamp_gain) * scale

            areas[l + 1].R.step(I_R)

    # ── full T-step rollout over a batch ─────────────────────
    def run_batch(self, pixel_pA: np.ndarray,
                  class_label: np.ndarray | None = None,
                  class_clamp_gain: float | None = None):
        """
        pixel_pA:    (B, n0) array of input currents (pA).
        class_label: (B,) int array or None.

        Returns (X_R, X_EP, X_EN), dicts of (B, n_l) time-window means.
        """
        pixel_pA = np.asarray(pixel_pA, dtype=np.float64)
        if pixel_pA.ndim != 2:
            raise ValueError(f"pixel_pA must be (B, n0); got shape {pixel_pA.shape}")
        B = pixel_pA.shape[0]

        if class_label is not None:
            class_label = np.asarray(class_label, dtype=np.int64).reshape(-1)
            if class_label.shape[0] != B:
                raise ValueError(
                    f"class_label batch dim {class_label.shape[0]} "
                    f"!= pixel_pA batch dim {B}")

        self.reset_all(B)
        n  = self.T
        tw = self.tw
        R_buf  = {l: [] for l in range(self.L)}
        EP_buf = {l: [] for l in range(self.L - 1)}
        EN_buf = {l: [] for l in range(self.L - 1)}

        for t in range(n):
            self.step(pixel_pA, class_label=class_label,
                      class_clamp_gain=class_clamp_gain)
            if t >= n - tw:
                for l in range(self.L):
                    R_buf[l].append(self.areas[l].R.X.copy())
                for l in range(self.L - 1):
                    EP_buf[l].append(self.areas[l].EP.X.copy())
                    EN_buf[l].append(self.areas[l].EN.X.copy())

        X_R  = {l: np.mean(R_buf[l],  axis=0) for l in range(self.L)}
        X_EP = {l: np.mean(EP_buf[l], axis=0) for l in range(self.L - 1)}
        X_EN = {l: np.mean(EN_buf[l], axis=0) for l in range(self.L - 1)}
        return X_R, X_EP, X_EN

    # ── single-sample convenience wrappers ───────────────────
    def run_sample_full(self, pixel_pA: np.ndarray,
                        class_label: int | None = None,
                        class_clamp_gain: float | None = None):
        pA_B = np.asarray(pixel_pA, dtype=np.float64)[None, :]
        lbl_B = np.array([int(class_label)], dtype=np.int64) if class_label is not None else None
        X_R, X_EP, X_EN = self.run_batch(pA_B, lbl_B, class_clamp_gain)
        X_R  = {l: X_R[l][0]  for l in range(self.L)}
        X_EP = {l: X_EP[l][0] for l in range(self.L - 1)}
        X_EN = {l: X_EN[l][0] for l in range(self.L - 1)}
        return X_R, X_EP, X_EN

    def predict_classes(self, pixel_pA: np.ndarray) -> np.ndarray:
        """pixel_pA: (B, n0). Returns (B,) int predictions."""
        if not self.use_class_area:
            raise ValueError("Classification area is disabled.")
        X_R, _, _ = self.run_batch(pixel_pA, class_label=None)
        return np.argmax(X_R[self.L - 1], axis=1).astype(int)

    def predict_class(self, pixel_pA: np.ndarray) -> int:
        return int(self.predict_classes(np.asarray(pixel_pA)[None, :])[0])

    # ── batched Hebbian update ───────────────────────────────
    def weight_update(self, X_R: dict, X_EP: dict, X_EN: dict):
        """
        X_R[l]:  (B, n_l)   for l in [0, L)
        X_EP[l]: (B, n_l)   for l in [0, L-1)
        X_EN[l]: (B, n_l)   for l in [0, L-1)
        """
        B = next(iter(X_R.values())).shape[0]
        for l in range(self.L - 1):
            ep = X_EP[l]              # (B, n_l)
            en = X_EN[l]              # (B, n_l)
            r  = X_R[l + 1]           # (B, n_l+1)
            dW_p = ep.T @ r / B       # (n_l, n_l+1)
            dW_n = en.T @ r / B
            g_W  = (self.areas[l].W > 0).astype(np.float64)
            dW   = self.lr * (dW_p - dW_n) - self.reg * g_W
            self.areas[l].W += dW
            np.maximum(self.areas[l].W, 0.0, out=self.areas[l].W)


# ─────────────────────────────────────────────────────────────
# 5.  Input Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess_image(img_flat: np.ndarray,
                     lo: float = 600.0,
                     hi: float = 3000.0) -> np.ndarray:
    x = img_flat.astype(np.float64) / 255.0
    n = np.linalg.norm(x)
    if n > 1e-10:
        x = x / n
    xmin, xmax = x.min(), x.max()
    if xmax > xmin:
        x = (x - xmin) / (xmax - xmin)
    return x * (hi - lo) + lo


def preprocess_batch(X: np.ndarray) -> np.ndarray:
    """X: (B, n) raw pixels in [0, 255]. Returns (B, n) pA, per-sample normalized."""
    return np.stack([preprocess_image(X[i]) for i in range(len(X))])


# ─────────────────────────────────────────────────────────────
# 6.  NRMSE — works for (n,) or (B, n) (last-axis reduction)
# ─────────────────────────────────────────────────────────────
def compute_nrmse(actual, predicted):
    actual    = np.asarray(actual)
    predicted = np.asarray(predicted)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2, axis=-1))
    r    = actual.max(axis=-1) - actual.min(axis=-1)
    if np.ndim(rmse) == 0:
        return 0.0 if r < 1e-10 else float(rmse / r)
    out = np.zeros_like(rmse, dtype=np.float64)
    mask = r > 1e-10
    out[mask] = rmse[mask] / r[mask]
    return out


# ─────────────────────────────────────────────────────────────
# 7.  Training Loop  (batched)
# ─────────────────────────────────────────────────────────────
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
        'nrmse'     : {l: [] for l in range(model.L - 1)},
        'cls_acc'   : [],
        'epoch_time': []
    }

    for epoch in range(n_epochs):
        t0       = time.time()
        perm     = rng.permutation(N)
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
                preds = np.argmax(X_R[model.L - 1], axis=1)
                ep_cls_acc.extend((preds == label_B).astype(float).tolist())

            for l in range(model.L - 1):
                # Reconstruction prediction at layer l from layer l+1: (B, n_l).
                pred_l = X_R[l + 1] @ model.areas[l].W.T
                ep_nrmse[l].extend(compute_nrmse(X_R[l], pred_l).tolist())

            model.weight_update(X_R, X_EP, X_EN)

        elapsed = time.time() - t0
        for l in range(model.L - 1):
            history['nrmse'][l].append(float(np.mean(ep_nrmse[l])))
        history['cls_acc'].append(
            float(np.mean(ep_cls_acc))
            if (use_classification and model.use_class_area and len(ep_cls_acc) > 0)
            else float('nan')
        )
        history['epoch_time'].append(elapsed)

        if verbose and (epoch % log_interval == 0 or epoch == n_epochs - 1):
            nstr = '  |  '.join(
                f"Area{l+1}: {history['nrmse'][l][-1]:.4f}"
                for l in range(model.L - 1))
            if use_classification and model.use_class_area:
                print(
                    f"  Epoch {epoch+1:3d}/{n_epochs}  |  {nstr}"
                    f"  |  ClsAcc: {history['cls_acc'][-1]:.3f}  |  {elapsed:.1f}s"
                )
            else:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}  |  {nstr}  |  {elapsed:.1f}s")
    return history


# ─────────────────────────────────────────────────────────────
# 8.  RSA Utilities
# ─────────────────────────────────────────────────────────────
# Spearman's rank correlation distance.
def compute_rdm(representations: np.ndarray) -> np.ndarray:
    Xr   = rankdata(representations, axis=1, method='average')
    X    = Xr - Xr.mean(axis=1, keepdims=True)
    nrm  = np.linalg.norm(X, axis=1, keepdims=True)
    nrm  = np.maximum(nrm, 1e-10)
    X    = X / nrm
    sim  = X @ X.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def second_order_rsa(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    N   = rdm_a.shape[0]
    idx = np.triu_indices(N, k=1)
    rho, _ = spearmanr(rdm_a[idx], rdm_b[idx])
    return float(rho)


# ─────────────────────────────────────────────────────────────
# 9.  Perturbation helpers
# ─────────────────────────────────────────────────────────────
def add_noise(X: np.ndarray, sigma_pA: float = 300.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    sigma_px = sigma_pA * 255.0 / 3000.0
    noise    = rng.normal(0.0, sigma_px, X.shape)
    return np.clip(X.astype(np.float64) + noise, 0.0, 255.0)


def add_occlusion(X: np.ndarray, patch_size: int = 9, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    out  = X.copy().astype(np.float64)
    imgs = out.reshape(-1, 28, 28)
    for img in imgs:
        r = rng.integers(0, 28 - patch_size)
        c = rng.integers(0, 28 - patch_size)
        img[r:r + patch_size, c:c + patch_size] = 0.0
    return imgs.reshape(X.shape)


# ─────────────────────────────────────────────────────────────
# 10. Evaluation helpers  (chunked batched)
# ─────────────────────────────────────────────────────────────
def get_representations(model: SNNPC, X: np.ndarray,
                        area: int = 1, verbose: bool = False,
                        chunk_size: int = 32):
    N = len(X)
    R_list = [None] * N
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pA_B = preprocess_batch(X[start:end])
        X_R, _, _ = model.run_batch(pA_B)
        for j in range(end - start):
            R_list[start + j] = {l: X_R[l][j].copy() for l in range(model.L)}
        if verbose:
            print(f"    {end}/{N}")
    R_mat = np.array([R_list[i][area] for i in range(N)])
    return R_mat, R_list


def linear_decode(R_tr, y_tr, R_te, y_te, random_state=None):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    clf.fit(R_tr, y_tr)
    return float(clf.score(R_te, y_te)), clf
