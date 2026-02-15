"""Train logistic model and save JSON-only coefficients."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import BASE_DIR, CURATED_DIR, DEFAULT_SEASON_CODE

FEATURES = ["d_r5_ts_pre", "d_r5_efg_pre", "d_r5_to_pre", "d_r5_margin_pre", "d_r5_usage_conc_pre", "home"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _fit_logreg_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 5000, l2: float = 1e-3):
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p_hat = _sigmoid(z)
        err = p_hat - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def run(season_code: str = DEFAULT_SEASON_CODE) -> dict:
    df = pd.read_csv(CURATED_DIR / "model_train.csv")
    if "season_code" in df.columns:
        df = df[df["season_code"] == season_code].copy()
    if df.empty:
        raise RuntimeError(f"No model training rows for season {season_code}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.sort_values(["date", "gamecode_num"], kind="stable").reset_index(drop=True)

    X = df[FEATURES].fillna(0.0).to_numpy(dtype=float)
    y = df["y"].astype(int).to_numpy(dtype=float)

    split_idx = max(int(len(df) * 0.8), 1)
    if split_idx >= len(df):
        split_idx = len(df) - 1
    if split_idx < 1:
        raise RuntimeError("Not enough rows for train/validation split")

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)

    X_train_s = (X_train - mean) / scale
    X_val_s = (X_val - mean) / scale

    coef, intercept = _fit_logreg_gd(X_train_s, y_train)
    p_val = _sigmoid(X_val_s @ coef + intercept)

    brier = float(np.mean((p_val - y_val) ** 2))
    eps = 1e-12
    ll = float(-np.mean(y_val * np.log(np.clip(p_val, eps, 1 - eps)) + (1 - y_val) * np.log(np.clip(1 - p_val, eps, 1 - eps))))
    print(f"Validation Brier: {brier:.4f}")
    print(f"Validation LogLoss: {ll:.4f}")

    payload = {
        "season_code": season_code,
        "feature_names": FEATURES,
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": float(intercept),
        "training_window": {
            "train_start": str(df.loc[0, "date"].date()) if pd.notna(df.loc[0, "date"]) else "",
            "train_end": str(df.loc[split_idx - 1, "date"].date()) if pd.notna(df.loc[split_idx - 1, "date"]) else "",
            "val_start": str(df.loc[split_idx, "date"].date()) if pd.notna(df.loc[split_idx, "date"]) else "",
            "val_end": str(df.loc[len(df) - 1, "date"].date()) if pd.notna(df.loc[len(df) - 1, "date"]) else "",
        },
        "metrics": {"brier": brier, "logloss": ll},
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }

    model_dir = BASE_DIR / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"win_model_{season_code}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train win model and save JSON coefficients")
    parser.add_argument("--season", dest="season_code", default=DEFAULT_SEASON_CODE)
    args = parser.parse_args()
    run(args.season_code)


if __name__ == "__main__":
    main()
