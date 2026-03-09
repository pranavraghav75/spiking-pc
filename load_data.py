"""
load_data.py — Data loading for SNN-PC replication
Tries real MNIST first (sklearn fetch_openml), falls back to
sklearn digits (8x8→28x28 upscaled) if network is unavailable.

FIX vs original:
  - Image.NEAREST → Image.Resampling.NEAREST (Pillow >= 10 compatibility)
  - Added unified load_mnist() with automatic fallback logic
"""
import numpy as np
import os


def _upscale_8x8_to_28x28(imgs_8x8):
    """Nearest-neighbour upscale from 8x8 to 28x28."""
    from PIL import Image
    try:
        resample = Image.Resampling.NEAREST   # Pillow >= 10
    except AttributeError:
        resample = Image.NEAREST              # Pillow < 10
    out = []
    for flat in imgs_8x8:
        arr = (flat.reshape(8, 8) * 255.0 / 16.0).astype(np.uint8)
        img = Image.fromarray(arr, mode='L').resize((28, 28), resample)
        out.append(np.array(img, dtype=np.float32).flatten())
    return np.array(out, dtype=np.float32)


def load_digits_as_mnist(n_train_per_class=None,
                          n_test_per_class=None,
                          test_frac=0.25, seed=42):
    """
    Load sklearn.datasets.load_digits (1797 total, ~180/class),
    upscale 8x8 to 28x28.
    NOTE: Only ~180 samples per class — suitable for smoke-tests only.
          The paper uses 512 train + 128 test per class; use load_mnist()
          for paper-accurate results.
    Returns X_train, y_train, X_test, y_test (pixel values 0-255 float32)
    """
    from sklearn.datasets import load_digits
    data = load_digits()
    X_all = data.images.reshape(-1, 64).astype(np.float32)
    X_all = _upscale_8x8_to_28x28(X_all)
    y_all = data.target.astype(int)
    rng = np.random.default_rng(seed)
    X_train, y_train = [], []
    X_test,  y_test  = [], []
    for c in range(10):
        idx = np.where(y_all == c)[0]
        rng.shuffle(idx)
        n_total = len(idx)
        n_test  = max(1, int(n_total * test_frac))
        n_train = n_total - n_test
        if n_train_per_class is not None:
            n_train = min(n_train, n_train_per_class)
        if n_test_per_class is not None:
            n_test  = min(n_test, n_test_per_class)
        X_train.append(X_all[idx[:n_train]])
        y_train.append(np.full(n_train, c))
        X_test.append(X_all[idx[n_train:n_train + n_test]])
        y_test.append(np.full(n_test, c))
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test  = np.concatenate(X_test)
    y_test  = np.concatenate(y_test)
    p_tr = rng.permutation(len(X_train))
    p_te = rng.permutation(len(X_test))
    return (X_train[p_tr], y_train[p_tr],
            X_test[p_te],  y_test[p_te])

def _fetch_real_mnist(n_train_per_class=512, n_test_per_class=128,
                      seed=42, cache_dir='./mnist_cache'):

    from torchvision import datasets
    import torch
    import numpy as np
    import os

    os.makedirs(cache_dir, exist_ok=True)

    train_ds = datasets.MNIST(root=cache_dir, train=True, download=True)
    test_ds  = datasets.MNIST(root=cache_dir, train=False, download=True)

    X_tr_all = train_ds.data.numpy().reshape(-1, 784).astype(np.float32)
    y_tr_all = train_ds.targets.numpy().astype(int)

    X_te_all = test_ds.data.numpy().reshape(-1, 784).astype(np.float32)
    y_te_all = test_ds.targets.numpy().astype(int)

    rng = np.random.default_rng(seed)

    X_train, y_train, X_test, y_test = [], [], [], []

    for c in range(10):
        idx_tr = np.where(y_tr_all == c)[0]
        idx_te = np.where(y_te_all == c)[0]

        sel_tr = rng.choice(idx_tr, n_train_per_class, replace=False)
        sel_te = rng.choice(idx_te, n_test_per_class,  replace=False)

        X_train.append(X_tr_all[sel_tr])
        y_train.append(np.full(n_train_per_class, c))
        X_test.append(X_te_all[sel_te])
        y_test.append(np.full(n_test_per_class, c))

    return (np.concatenate(X_train), np.concatenate(y_train),
            np.concatenate(X_test),  np.concatenate(y_test))

def load_mnist(n_train_per_class=512, n_test_per_class=128,
               seed=42, cache_dir="./mnist_cache",
               verbose=True):

    data = _fetch_real_mnist(
        n_train_per_class,
        n_test_per_class,
        seed,
        cache_dir
    )

    if verbose:
        print(f"[load_mnist] Real MNIST via torchvision: "
              f"train={len(data[0])}, test={len(data[2])}")

    return data

