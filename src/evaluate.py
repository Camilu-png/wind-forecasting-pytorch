import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        warnings.warn("R² is undefined: target values are nearly constant.")
        r2 = float("nan")
    else:
        r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list = []
    all_targets: list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()
    return y_true, y_pred


def plot_loss_curves(
    train_losses: Dict[int, float],
    val_losses: Dict[int, float],
    save_path: str,
) -> None:
    epochs = sorted(train_losses.keys())
    train_vals = [train_losses[e] for e in epochs]
    val_vals = [val_losses[e] for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_vals, label="Train", linewidth=1.5)
    ax.plot(epochs, val_vals, label="Validation", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    n_samples: int = 500,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.4, s=8)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    axes[0].set_xlabel("True Power")
    axes[0].set_ylabel("Predicted Power")
    axes[0].set_title("Predictions vs Ground Truth")
    axes[0].grid(True, alpha=0.3)

    n = min(n_samples, len(y_true))
    indices = np.arange(n)
    axes[1].plot(indices, y_true[:n], label="True", linewidth=1.0, alpha=0.8)
    axes[1].plot(indices, y_pred[:n], label="Predicted", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Power")
    axes[1].set_title(f"Predictions Over Test Samples (first {n})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=8)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Power")
    axes[0].set_ylabel("Residual (True - Predicted)")
    axes[0].set_title("Residual Plot")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=80, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
