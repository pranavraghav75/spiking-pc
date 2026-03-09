"""
run_experiment.py — Full experiment runner for Lee et al. (2024) SNN-PC replication.

Reproduces:
  Figure 6 — NRMSE learning curves + RDMs + second-order RSA
  Figure 7 — Robustness: Clean / Noise / Occlude
  Figure 8 — FFG ablation: PC+FFG vs PC-only vs FFG-only

Usage:
  python run_experiment.py              # full paper setup
  python run_experiment.py --smoke      # fast smoke-test
  python run_experiment.py --no-fig8   # skip ablation
"""

import argparse, os, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from snn_pc import (SNNPC, train_snn_pc, preprocess_image,
                    compute_rdm, second_order_rsa,
                    get_representations, linear_decode,
                    add_noise, add_occlusion, compute_nrmse)
from load_data import load_mnist


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def reconstruct_from_area1(model, X_R_dict):
    """pred_0 = W[0] @ X_R[1]"""
    return model.areas[0].W @ X_R_dict[1]


def plot_reconstructions(model, X_samples, y_samples, R_list,
                         title, outpath):
    classes = sorted(set(y_samples))
    n_cls   = len(classes)
    fig, axes = plt.subplots(2, n_cls, figsize=(n_cls * 1.1, 2.6))
    if n_cls == 1:
        axes = axes.reshape(2, 1)
    for ci, c in enumerate(classes):
        idx = np.where(np.array(y_samples) == c)[0]
        if len(idx) == 0:
            continue
        i   = idx[0]
        inp = np.array(X_samples[i]).reshape(28, 28)
        rec = reconstruct_from_area1(model, R_list[i]).reshape(28, 28)
        axes[0, ci].imshow(inp, cmap='gray', vmin=0, vmax=255)
        axes[0, ci].axis('off')
        axes[0, ci].set_title(str(c), fontsize=7)
        axes[1, ci].imshow(rec, cmap='gray')
        axes[1, ci].axis('off')
    axes[0, 0].set_ylabel('Input', fontsize=7)
    axes[1, 0].set_ylabel('Recon', fontsize=7)
    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_nrmse_curves(history, outpath):
    n = len(history['nrmse'])
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3))
    if n == 1:
        axes = [axes]
    for l, ax in enumerate(axes):
        ax.plot(range(1, len(history['nrmse'][l]) + 1),
                history['nrmse'][l], lw=1.5, color='steelblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NRMSE')
        ax.set_title(f'Area {l+1}')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Prediction Error During Training', fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def eval_rsa_and_decode(model, X_eval, y_eval, X_train_rep, y_train_rep,
                        n_per_class=None, rng=None):
    """
    Returns: rho (RSA vs input), decode_acc
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_per_class is not None:
        idx = []
        for c in sorted(set(y_eval)):
            ci = np.where(y_eval == c)[0]
            sel = rng.choice(ci, min(n_per_class, len(ci)), replace=False)
            idx.extend(sel)
        X_eval = X_eval[np.array(idx)]
        y_eval = y_eval[np.array(idx)]

    pA_all  = np.array([preprocess_image(X_eval[i]) for i in range(len(X_eval))])
    rdm_inp = compute_rdm(pA_all)

    _, R_list = get_representations(model, X_eval, area=1)
    R_mat = np.array([R_list[i][1] for i in range(len(X_eval))])
    rdm_r = compute_rdm(R_mat)
    rho   = second_order_rsa(rdm_inp, rdm_r)

    R_tr_full, _ = get_representations(model, X_train_rep, area=1)
    acc, _ = linear_decode(R_tr_full, y_train_rep, R_mat, y_eval)

    return rho, acc, R_list, rdm_inp, rdm_r


# ─────────────────────────────────────────────────────────────
# Figure 6
# ─────────────────────────────────────────────────────────────
def run_figure6(model, X_train, y_train, X_test, y_test,
                history, outdir, n_rsa_per_class=12):
    print("\n── Figure 6: Representational Learning ──")
    os.makedirs(outdir, exist_ok=True)

    plot_nrmse_curves(history, os.path.join(outdir, 'fig6B_nrmse.png'))

    # Reconstruct test samples
    _, R_list_te = get_representations(model, X_test, verbose=True)
    plot_reconstructions(
        model, X_test, y_test, R_list_te,
        title='Fig 6A — Reconstruction',
        outpath=os.path.join(outdir, 'fig6A_recon.png'))

    # RSA
    rng = np.random.default_rng(7)
    idx = []
    for c in sorted(set(y_test)):
        ci = np.where(y_test == c)[0]
        sel = rng.choice(ci, min(n_rsa_per_class, len(ci)), replace=False)
        idx.extend(sel)
    idx = np.array(idx)
    X_rsa, y_rsa = X_test[idx], y_test[idx]
    pA_all  = np.array([preprocess_image(X_rsa[i]) for i in range(len(X_rsa))])
    rdm_inp = compute_rdm(pA_all)

    rhos = {}
    for l in range(1, model.L):
        R_mat = np.array([R_list_te[idx[j]][l] for j in range(len(idx))])
        rdm_l = compute_rdm(R_mat)
        rhos[l] = second_order_rsa(rdm_inp, rdm_l)

        fig, ax = plt.subplots(figsize=(3, 2.8))
        im = ax.imshow(rdm_l, cmap='RdBu_r', vmin=0, vmax=2)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'RDM Area {l}', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig6C_rdm_area{l}.png'), dpi=120)
        plt.close()

    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.barh([f'Area {l}' for l in rhos], list(rhos.values()), color='steelblue')
    ax.set_xlabel('Representational similarity (ρ)')
    ax.set_xlim(0, 1.05)
    ax.set_title('Fig 6D — 2nd-order RSA')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig6D_rsa.png'), dpi=120)
    plt.close()
    print(f"  RSA rhos (area→rho): {rhos}")
    return rhos


# ─────────────────────────────────────────────────────────────
# Figure 7
# ─────────────────────────────────────────────────────────────
def run_figure7(model, X_train, y_train, X_test, y_test, outdir,
                n_per_class=12, n_rep=5):
    print("\n── Figure 7: Robustness ──")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(2)

    rho_results = {k: [] for k in ('Clean', 'Noise', 'Occlude')}
    acc_results = {k: [] for k in ('Clean', 'Noise', 'Occlude')}
    example     = {}

    for rep in range(n_rep):
        idx = []
        for c in sorted(set(y_test)):
            ci  = np.where(y_test == c)[0]
            sel = rng.choice(ci, min(n_per_class, len(ci)), replace=False)
            idx.extend(sel)
        idx = np.array(idx)
        X_te, y_te = X_test[idx], y_test[idx]
        X_no = add_noise(X_te, sigma_pA=300., rng=rng)
        X_oc = add_occlusion(X_te, patch_size=9, rng=rng)

        for name, X_set in [('Clean', X_te), ('Noise', X_no), ('Occlude', X_oc)]:
            rho, acc, R_list, _, _ = eval_rsa_and_decode(
                model, X_set, y_te, X_train, y_train, rng=rng)
            rho_results[name].append(rho)
            acc_results[name].append(acc)
            if rep == 0:
                example[name] = (X_set, y_te, R_list)

    for name, (X_s, y_s, R_l) in example.items():
        plot_reconstructions(
            model, X_s, y_s, R_l,
            title=f'Fig 7A — {name}',
            outpath=os.path.join(outdir, f'fig7A_{name}.png'))

    conditions = ['Clean', 'Noise', 'Occlude']
    colors = ['steelblue', 'salmon', 'mediumpurple']

    for results, ylabel, fname, fig_label in [
        (rho_results, 'Representational similarity (ρ)', 'fig7C_rsa.png', 'C'),
        (acc_results, 'Decoding accuracy',               'fig7D_dec.png', 'D'),
    ]:
        means = [np.mean(results[c]) for c in conditions]
        sems  = [np.std(results[c]) / np.sqrt(max(len(results[c]), 1))
                 for c in conditions]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(conditions, means, yerr=sems, capsize=4, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Fig 7{fig_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=120)
        plt.close()
        summary = ", ".join(f"{c}={np.mean(results[c]):.3f}" for c in conditions)
        print(f"  {ylabel}: {summary}")

    return rho_results, acc_results


# ─────────────────────────────────────────────────────────────
# Figure 8
# ─────────────────────────────────────────────────────────────
def run_figure8(X_train, y_train, X_test, y_test, outdir,
                n_epochs, batch_size, n_per_class=12, verbose=True):
    print("\n── Figure 8: FFG Ablation ──")
    os.makedirs(outdir, exist_ok=True)

    trained = {}
    for name, use_ffg in [('PC+FFG', True), ('PC-only', False)]:
        print(f"\n  Training {name}...")
        m = SNNPC(use_ffg=use_ffg, lr=1e-5, reg=1e-7, rng=np.random.default_rng(42))
        h = train_snn_pc(m, X_train, y_train, n_epochs=n_epochs,
                         batch_size=batch_size, verbose=verbose)
        trained[name] = m

    # FFG-only: zero out PC weights on a fresh PC+FFG model
    print("\n  Building FFG-only (W=0) model...")
    m_fgo = SNNPC(use_ffg=True, rng=np.random.default_rng(42))
    for l in range(m_fgo.L - 1):
        m_fgo.areas[l].W[:] = 0.0
    trained['FFG-only'] = m_fgo

    # Evaluation set
    rng = np.random.default_rng(11)
    idx = []
    for c in sorted(set(y_test)):
        ci  = np.where(y_test == c)[0]
        sel = rng.choice(ci, min(n_per_class, len(ci)), replace=False)
        idx.extend(sel)
    idx = np.array(idx)
    X_ev, y_ev = X_test[idx], y_test[idx]

    pA_all  = np.array([preprocess_image(X_ev[i]) for i in range(len(X_ev))])
    rdm_inp = compute_rdm(pA_all)

    rhos, accs = {}, {}
    for name, m in trained.items():
        _, R_list = get_representations(m, X_ev, area=1)
        R_mat = np.array([R_list[i][1] for i in range(len(X_ev))])
        rdm_r = compute_rdm(R_mat)
        rhos[name] = second_order_rsa(rdm_inp, rdm_r)

        R_tr, _ = get_representations(m, X_train, area=1)
        acc, _  = linear_decode(R_tr, y_train, R_mat, y_ev)
        accs[name] = acc

        _, R_list_all = get_representations(m, X_test, area=1)
        plot_reconstructions(m, X_test, y_test, R_list_all,
                             title=f'Fig 8A — {name}',
                             outpath=os.path.join(outdir,
                                                  f'fig8A_{name}.png'))

    print(f"  RSA: {rhos}")
    print(f"  Acc: {accs}")

    all_names  = ['Input', 'PC+FFG', 'PC-only', 'FFG-only']
    all_colors = ['gray', 'steelblue', 'salmon', 'mediumpurple']
    acc_inp, _ = linear_decode(pA_all, y_ev, pA_all, y_ev)

    for values_d, inp_val, ylabel, fname in [
        (rhos, 1.0,     'Representational similarity (ρ)', 'fig8C.png'),
        (accs, acc_inp, 'Decoding accuracy',               'fig8D.png'),
    ]:
        vals = [inp_val] + [values_d[k] for k in ('PC+FFG', 'PC-only', 'FFG-only')]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(all_names, vals, color=all_colors)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=120)
        plt.close()
        print(f"  Saved: {os.path.join(outdir, fname)}")

    return trained, rhos, accs


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',   action='store_true')
    parser.add_argument('--outdir',  default='results')
    parser.add_argument('--no-fig8', dest='no_fig8', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.smoke:
        N_TR, N_TE, N_EP, BS, NRSA, NREP = 5, 5, 2, 5, 5, 2
        print("[SMOKE] Minimal run — not paper-accurate.")
    else:
        N_TR, N_TE, N_EP, BS, NRSA, NREP = 512, 128, 50, 32, 12, 5

    # ── Data ─────────────────────────────────────────────────
    print("\n=== Loading data ===")
    X_tr, y_tr, X_te, y_te = load_mnist(n_train_per_class=N_TR,
                                          n_test_per_class=N_TE)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    # ── Train PC+FFG ─────────────────────────────────────────
    print("\n=== Training PC+FFG ===")
    # lr=1e-5, reg=1e-7: learning signal dominates regularization from epoch 1.
    # Paper's lr=1e-7/reg=1e-5 was calibrated for large late-training traces;
    # at initialization the reg term is 35x stronger and erases weights.
    model = SNNPC(use_ffg=True, lr=1e-5, reg=1e-7, rng=np.random.default_rng(42))
    history = train_snn_pc(model, X_tr, y_tr,
                           n_epochs=N_EP, batch_size=BS,
                           verbose=True, log_interval=1)

    np.savez(os.path.join(args.outdir, 'weights.npz'),
             **{f'W{l}': model.areas[l].W for l in range(model.L - 1)})

    # ── Figures ──────────────────────────────────────────────
    run_figure6(model, X_tr, y_tr, X_te, y_te, history,
                outdir=os.path.join(args.outdir, 'fig6'),
                n_rsa_per_class=NRSA)

    run_figure7(model, X_tr, y_tr, X_te, y_te,
                outdir=os.path.join(args.outdir, 'fig7'),
                n_per_class=NRSA, n_rep=NREP)

    if not args.no_fig8:
        run_figure8(X_tr, y_tr, X_te, y_te,
                    outdir=os.path.join(args.outdir, 'fig8'),
                    n_epochs=N_EP, batch_size=BS,
                    n_per_class=NRSA)

    print(f"\n=== Done. Results in: {args.outdir}/ ===")


if __name__ == '__main__':
    main()
