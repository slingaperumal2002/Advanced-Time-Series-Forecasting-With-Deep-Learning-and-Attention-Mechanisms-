"""Data pipeline:
- Can generate synthetic multivariate time series with trend + seasonality + noise
- Or load a preprocessed CSV (user can adapt to M4)
- Outputs numpy arrays saved under ./data/
"""
import argparse
import numpy as np
import os
import json

def generate_synthetic(n_series=3, length=1500, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(length)
    data = []
    for i in range(n_series):
        trend = 0.001 * t * (1 + 0.5 * rng.randn())
        seasonal = 2.0 * np.sin(2 * np.pi * t / (50 + rng.randint(0,100))) 
        noise = 0.5 * rng.randn(length)
        series = trend + seasonal + noise + rng.randn()*0.1
        data.append(series)
    return np.stack(data, axis=1)  # shape (length, n_series)

def save_data(arr, out_dir="./data", name="synth.npy"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    np.save(path, arr)
    print(f"Saved data to {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synth", "load"], default="synth")
    parser.add_argument("--n_series", type=int, default=3)
    parser.add_argument("--length", type=int, default=1500)
    parser.add_argument("--out", type=str, default="./data/synth.npy")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    if args.mode == 'synth':
        data = generate_synthetic(n_series=args.n_series, length=args.length)
        save_data(data, out_dir=os.path.dirname(args.out) or './data', name=os.path.basename(args.out))
    else:
        raise NotImplementedError("CSV loader not implemented; adapt for M4 dataset.")
