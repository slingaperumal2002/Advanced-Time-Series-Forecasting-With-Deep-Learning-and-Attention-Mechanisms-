# Advanced Time Series Forecasting with Attention (Boilerplate v2)

This boilerplate addresses the AI evaluation feedback by providing:
- A synthetic multivariate time series generator (data pipeline)
- A PyTorch Seq2Seq + Attention model implementation
- Training script with Optuna-based hyperparameter search (Bayesian optimization)
- Walk-forward evaluation utilities and standard metrics (RMSE, MAE, MAPE)
- SHAP explainability integration hooks (requires `shap` to run)
- Example Jupyter notebook with end-to-end pipeline

**How to run**
1. Create a virtual environment and install requirements (see `requirements.txt`).
2. Generate data: `python src/data_pipeline.py --mode synth --n_series 5 --length 2000`
3. Run a quick train: `python src/train.py --config configs/default.json`
4. Evaluate: `python src/evaluate.py --model_path models/best_model.pt`
5. Open `notebooks/analysis.ipynb` for exploratory analysis and SHAP.
