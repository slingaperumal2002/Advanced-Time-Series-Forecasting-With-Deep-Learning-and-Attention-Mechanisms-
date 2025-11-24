"""Evaluation script: loads model, computes RMSE, MAE, MAPE and saves outputs.
Also saves attention maps for inspection.
"""
import numpy as np
import torch
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(a,b): return np.sqrt(mean_squared_error(a.reshape(-1), b.reshape(-1)))
def mape(a,b): return np.mean(np.abs((a-b)/ (np.clip(a,1e-6,None))))*100

def main(model_path='models/best_model.pt', data_path='data/synth.npy'):
    arr = np.load(data_path)
    cfg = json.load(open('configs/default.json'))
    from model import Seq2Seq
    device = torch.device(cfg.get('device','cpu'))
    # quick eval on last window
    input_window = cfg['input_window']
    horizon = cfg['forecast_horizon']
    x = arr[-(input_window+horizon):-horizon][None,:,:]
    y_true = arr[-horizon:][None,:,:]
    import torch
    model = Seq2Seq(input_size=x.shape[2], output_size=y_true.shape[2], hidden_size=cfg['hidden_size'], num_layers=cfg.get('num_layers',1))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    xb = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred, attn = model(xb, horizon, teacher_forcing_ratio=0.0)
    pred = pred.cpu().numpy()
    attn = attn.cpu().numpy()
    metrics = {
        'rmse': float(rmse(pred, y_true)),
        'mae': float(mean_absolute_error(y_true.reshape(-1), pred.reshape(-1))),
        'mape': float(mape(y_true, pred))
    }
    print('Metrics:', metrics)
    np.save('models/last_pred.npy', pred)
    np.save('models/last_attn.npy', attn)
    with open('models/metrics.json','w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
