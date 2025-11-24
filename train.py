"""Training script with Optuna hyperparameter search hook.
- Provides a quick-run mode and an Optuna study for Bayesian optimization.
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import optuna
import os
from model import Seq2Seq

def load_data(path='./data/synth.npy'):
    arr = np.load(path)  # shape (length, n_series)
    return arr

def prepare_windows(arr, input_window=64, horizon=24):
    X, Y = [], []
    L, D = arr.shape
    for i in range(L - input_window - horizon + 1):
        x = arr[i:i+input_window]
        y = arr[i+input_window:i+input_window+horizon]
        X.append(x)
        Y.append(y)
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

def train_one(cfg, trial=None):
    cfg_local = cfg.copy()
    if trial:
        cfg_local['hidden_size'] = trial.suggest_categorical('hidden_size', [64, 128, 256])
        cfg_local['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    arr = load_data('./data/synth.npy')
    X, Y = prepare_windows(arr, cfg_local['input_window'], cfg_local['forecast_horizon'])
    device = torch.device(cfg_local.get('device','cpu'))
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=cfg_local['batch_size'], shuffle=True)
    model = Seq2Seq(input_size=X.shape[2], output_size=Y.shape[2], hidden_size=cfg_local['hidden_size'], num_layers=cfg_local.get('num_layers',1), device=str(device)).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg_local['lr'])
    loss_fn = nn.MSELoss()
    best = 1e9
    for epoch in range(cfg_local['epochs']):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred, _ = model(xb, cfg_local['forecast_horizon'])
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(loader.dataset)
        if avg < best:
            best = avg
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pt')
        print(f"Epoch {epoch+1}/{cfg_local['epochs']} loss={avg:.6f}")
    return best

def objective(trial):
    cfg = json.load(open('configs/default.json'))
    return train_one(cfg, trial=trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--optuna', action='store_true')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    if args.optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        print('Best trial:', study.best_trial.params)
    else:
        train_one(cfg)
