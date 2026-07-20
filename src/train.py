import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Tuple, Dict


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    checkpoint_dir: str = "outputs/trained_models",
    model_name: str = "best_model",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[int, float], Dict[int, float]]:
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_losses: Dict[int, float] = {}
    val_losses: Dict[int, float] = {}

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if (epoch % 10 == 0) or (epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"Train Loss: {train_loss:.6f}  "
                f"Val Loss: {val_loss:.6f}  "
                f"LR: {lr:.2e}  "
                f"Best: {best_epoch} ({best_val_loss:.6f})"
            )

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    print(f"  Training complete. Best epoch: {best_epoch}, Val Loss: {best_val_loss:.6f}")

    model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, f"{model_name}.pt"), weights_only=True)
    )
    return train_losses, val_losses
