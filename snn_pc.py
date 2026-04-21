"""
snn_pc.py — Spiking Neural Network for Predictive Coding
Faithful replication of Lee et al. (2024)
Frontiers in Computational Neuroscience 18:1338280

Architecture
  Area 0 (784)  →  Area 1 (400)  →  Area 2 (225)  →  Area 3 (64)
  Each area l>0: R^l (representation), E+^l, E-^l (error units)
  FFG pathway:   R^0 → G (16 gist units) → R^l  [fixed, sparse]
  Learning:      Hebbian on inter-areal weights W^{l,l+1}
  Neurons:       AdEx (adaptive exponential integrate-and-fire)
  Synapses:      AMPA + NMDA kinetics (rise 5 ms, decay 50 ms)

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
from scipy.stats import spearmanr
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
# 1.  AdEx Neuron Group
# ─────────────────────────────────────────────────────────────
class NeuronGroup:
    def __init__(self, n: int, dt: float = 1e-3):
        self.n   = n
        self.dt  = dt
        self.V   = np.full(n, EL, dtype=np.float64)
        self.a   = np.zeros(n, dtype=np.float64)
        self.Y   = np.zeros(n, dtype=np.float64)
        self.X   = np.zeros(n, dtype=np.float64)
        self.spk = np.zeros(n, dtype=bool)
        self.ref = np.zeros(n, dtype=int)

    def step(self, I_ext: np.ndarray):
        dt = self.dt
        V  = self.V
        a  = self.a
        exp_arg  = np.clip((V - VT) / DELTA_T, -30.0, 20.0) # remove clip function here
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
        self.X  = np.maximum(self.X, 0.0)
        self.Y  = np.maximum(self.Y, 0.0) 

    def reset(self):
        self.V[:]   = EL
        self.a[:]   = 0.0
        self.Y[:]   = 0.0
        self.X[:]   = 0.0
        self.spk[:] = False
        self.ref[:] = 0


# ─────────────────────────────────────────────────────────────
# 2.  Feedforward Gist (FFG) Pathway
# ─────────────────────────────────────────────────────────────
class FFGPathway:
    def __init__(self, n_input: int, n_gist: int,
                 area_sizes: list, pc: float = 0.05, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.n_gist = n_gist
        self.G = NeuronGroup(n_gist)
        ratio_ig = n_input / n_gist
        mask_ig  = (rng.random((n_input, n_gist)) < pc).astype(np.float64)
        self.W_IG = rng.normal(ratio_ig, ratio_ig, (n_input, n_gist)) * mask_ig
        self.W_GR = {}
        for l, ns in enumerate(area_sizes[1:], start=1):
            ratio_gr = n_gist / ns
            self.W_GR[l] = rng.normal(ratio_gr, ratio_gr, (n_gist, ns))

    def step(self, X_R0: np.ndarray, dt: float):
        I_G = (self.W_IG.T @ X_R0) * 1e-12
        self.G.step(I_G)

    def gist_input(self, l: int) -> np.ndarray:
        return self.W_GR[l].T @ self.G.X

    def reset(self):
        self.G.reset()


# ─────────────────────────────────────────────────────────────
# 3.  Predictive-Coding Area
# ─────────────────────────────────────────────────────────────
class PCArea:
    def __init__(self, l: int, n_R: int, n_R_above=None, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.l   = l
        self.n_R = n_R
        self.R   = NeuronGroup(n_R)
        self.EP  = NeuronGroup(n_R)
        self.EN  = NeuronGroup(n_R)
        if n_R_above is not None:
            # FIX: std = 1.0/n_R_above  (original was 0.3/n_R_above, sub-threshold)
            scaled_std = 1.0 / n_R_above
            raw = rng.normal(0.0, scaled_std, (n_R, n_R_above))
            self.W = np.abs(raw)
        else:
            self.W = None

    def reset(self):
        self.R.reset()
        self.EP.reset()
        self.EN.reset()


# ─────────────────────────────────────────────────────────────
# 4.  Full SNN-PC Network
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
            rng = np.random.default_rng(42)

        self.area_sizes = area_sizes
        self.L          = len(area_sizes)
        self.dt         = dt
        self.lr         = lr
        self.reg        = reg
        self.tw         = int(tw_ms / (dt * 1000))
        self.T          = int(T_ms  / (dt * 1000))
        self.use_ffg    = use_ffg
        self.I_scale    = 1e-12
        self.syn_gain   = syn_gain
        self.use_class_area = use_class_area
        self.n_classes = n_classes
        self.cls_clamp_gain = cls_clamp_gain
        self.areas = []
        for l in range(self.L):
            n_above = area_sizes[l + 1] if l < self.L - 1 else None
            self.areas.append(PCArea(l, area_sizes[l], n_above, rng=rng))
        if use_ffg:
            self.ffg = FFGPathway(area_sizes[0], n_gist, area_sizes, rng=rng)

    def label_clamp_current(self, label: int, gain: float | None = None) -> np.ndarray:
        if not self.use_class_area:
            raise ValueError("Classification area is disabled.")
        if gain is None:
            gain = self.cls_clamp_gain
        clamp = np.zeros(self.area_sizes[-1], dtype=np.float64)
        clamp[label] = gain
        return clamp

    def predict_class(self, pixel_pA: np.ndarray) -> int:
        if not self.use_class_area:
            raise ValueError("Classification area is disabled.")
        X_R, _, _ = self.run_sample_full(pixel_pA, class_label=None)
        return int(np.argmax(X_R[self.L - 1]))

    def reset_all(self):
        for area in self.areas:
            area.reset()
        if self.use_ffg:
            self.ffg.reset()

    # def step(self, pixel_pA: np.ndarray):
    #     areas = self.areas
    #     dt    = self.dt
    #     gain  = self.syn_gain # figure out, dont know what this is
    #     scale = self.I_scale # figure out, dont know what this is
    #     if self.use_ffg:
    #         self.ffg.step(areas[0].R.X, dt)
        
    #     # starting at R0, then update E0+ and E0-, then update R1 and E1+ and E1-
    #     for l in range(self.L - 1):
    #         X_R_below = areas[l].R.X
    #         pred = areas[l].W @ areas[l + 1].R.X
    #         # compute spike trace layer by layer, use to update next level
    #         areas[l].EP.step((X_R_below - pred) * gain * scale)
    #         areas[l].EN.step((pred - X_R_below) * gain * scale)
    #         # areas[l].EP.step(X_R_below - pred)
    #         # areas[l].EN.step(pred - X_R_below)
    #     areas[0].R.step(pixel_pA * scale)
    #     # areas[0].R.step(pixel_pA)
    #     for l in range(1, self.L):
    #         W_below = areas[l - 1].W
    #         bu_p = W_below.T @ areas[l - 1].EP.X
    #         bu_n = W_below.T @ areas[l - 1].EN.X
    #         td_p = areas[l].EP.X
    #         td_n = areas[l].EN.X
    #         I_R = (bu_p - bu_n - td_p + td_n) * gain * scale
    #         # I_R = (bu_p - bu_n - td_p + td_n)
    #         if self.use_ffg:
    #             I_R = I_R + self.ffg.gist_input(l) * gain * scale
    #             # I_R = I_R + self.ffg.gist_input(l)
    #         areas[l].R.step(I_R)

# take out gain from all functions
    def step(self, pixel_pA: np.ndarray, class_label: int | None = None,
             class_clamp_gain: float | None = None):
        areas = self.areas
        dt = self.dt
        gain = self.syn_gain
        scale = self.I_scale

        # if self.use_ffg:
        #     self.ffg.step(areas[0].R.X, dt)

        areas[0].R.step(pixel_pA * scale)
        td_p = 0.0
        td_n = 0.0

        for l in range(self.L - 1):
            pred = areas[l].W @ areas[l + 1].R.X
            X_R_current = areas[l].R.X
            
            areas[l].EP.step((X_R_current - pred) * gain * scale)
            areas[l].EN.step((pred - X_R_current) * gain * scale)

            W_below = areas[l].W
            bu_p = W_below.T @ areas[l].EP.X
            bu_n = W_below.T @ areas[l].EN.X


            if l + 1 < self.L - 1:
                td_p = areas[l + 1].EP.X
                td_n = areas[l + 1].EN.X
            else:
                td_p = 0.0
                td_n = 0.0

            I_R = (bu_p - bu_n - td_p + td_n) * gain * scale

            # if self.use_ffg:
            #     I_R = I_R + self.ffg.gist_input(l + 1) * gain * scale

            # Optional supervised label clamp on the top class area.
            if self.use_class_area and class_label is not None and (l + 1) == (self.L - 1):
                I_R = I_R + self.label_clamp_current(class_label, class_clamp_gain) * scale

            areas[l + 1].R.step(I_R)
            # make I_R equal to arbitrarily high number, and set correct neuron in last area of neurons equal to I_R, everything else 0 and take out label_clamp_current
            # fired = (self.V > VT) & (~in_ref), make fired (in nueron class) equal to True to intentioally control spike rate (determine some k for how many iterations we set fired equal to true)
 

    def run_sample_full(self, pixel_pA: np.ndarray, class_label: int | None = None,
                        class_clamp_gain: float | None = None):
        self.reset_all()
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

    def weight_update(self, X_R_batch, X_EP_batch, X_EN_batch):
        B = len(X_R_batch)
        for l in range(self.L - 1):
            dW_p = np.zeros_like(self.areas[l].W)
            dW_n = np.zeros_like(self.areas[l].W)
            for i in range(B):
                ep = X_EP_batch[i][l]
                en = X_EN_batch[i][l]
                r  = X_R_batch[i][l + 1]
                dW_p += np.outer(ep, r)
                dW_n += np.outer(en, r)
            dW_p /= B
            dW_n /= B
            g_W = (self.areas[l].W > 0).astype(np.float64)
            dW  = self.lr * (dW_p - dW_n) - self.reg * g_W
            self.areas[l].W += dW
            self.areas[l].W  = np.maximum(self.areas[l].W, 0.0)


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


# ─────────────────────────────────────────────────────────────
# 6.  Training Loop
# ─────────────────────────────────────────────────────────────
def compute_nrmse(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    r    = actual.max() - actual.min()
    return 0.0 if r < 1e-10 else rmse / r


def train_snn_pc(model: SNNPC, X_train, y_train,
                 n_epochs=50, batch_size=32,
                 verbose=True, log_interval=5,
                 use_classification=False,
                 class_clamp_gain: float | None = None):
    N = len(X_train)
    n_batches = N // batch_size
    rng = np.random.default_rng(0)
    history = {
        'nrmse'     : {l: [] for l in range(model.L - 1)},
        'cls_acc'   : [],
        'epoch_time': []
    }
    for epoch in range(n_epochs):
        t0      = time.time()
        perm    = rng.permutation(N)
        ep_nrmse = {l: [] for l in range(model.L - 1)}
        ep_cls_acc = []
        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            X_R_b, X_EP_b, X_EN_b = [], [], []
            for i in idx:
                pA = preprocess_image(X_train[i])
                label = int(y_train[i]) if (use_classification and model.use_class_area) else None
                X_R, X_EP, X_EN = model.run_sample_full(
                    pA,
                    class_label=label,
                    class_clamp_gain=class_clamp_gain,
                )
                X_R_b.append(X_R)
                X_EP_b.append(X_EP)
                X_EN_b.append(X_EN)
                if use_classification and model.use_class_area:
                    pred = int(np.argmax(X_R[model.L - 1]))
                    ep_cls_acc.append(1.0 if pred == int(y_train[i]) else 0.0)
                for l in range(model.L - 1):
                    pred = model.areas[l].W @ X_R[l + 1]
                    ep_nrmse[l].append(compute_nrmse(X_R[l], pred))
            model.weight_update(X_R_b, X_EP_b, X_EN_b)
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
# 7.  RSA Utilities
# ─────────────────────────────────────────────────────────────
def compute_rdm(representations: np.ndarray) -> np.ndarray:
    X    = representations - representations.mean(axis=1, keepdims=True)
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
# 8.  Perturbation helpers
# ─────────────────────────────────────────────────────────────
def add_noise(X: np.ndarray, sigma_pA: float = 300.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    sigma_px = sigma_pA * 255.0 / 3000.0
    noise    = rng.normal(0.0, sigma_px, X.shape)
    return np.clip(X.astype(np.float64) + noise, 0.0, 255.0)


def add_occlusion(X: np.ndarray, patch_size: int = 9, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    out  = X.copy().astype(np.float64)
    imgs = out.reshape(-1, 28, 28)
    for img in imgs:
        r = rng.integers(0, 28 - patch_size)
        c = rng.integers(0, 28 - patch_size)
        img[r:r + patch_size, c:c + patch_size] = 0.0
    return imgs.reshape(X.shape)


# ─────────────────────────────────────────────────────────────
# 9.  Evaluation helpers
# ─────────────────────────────────────────────────────────────
def get_representations(model: SNNPC, X: np.ndarray,
                        area: int = 1, verbose: bool = False):
    N      = len(X)
    R_list = []
    for i in range(N):
        pA = preprocess_image(X[i])
        X_R, _, _ = model.run_sample_full(pA)
        R_list.append({l: X_R[l].copy() for l in range(model.L)})
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{N}")
    R_mat = np.array([R_list[i][area] for i in range(N)])
    return R_mat, R_list


def linear_decode(R_tr, y_tr, R_te, y_te):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=0)
    clf.fit(R_tr, y_tr)
    return float(clf.score(R_te, y_te)), clf