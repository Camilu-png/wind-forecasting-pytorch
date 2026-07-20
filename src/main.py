import os
import pickle
import argparse
import torch
import pandas as pd
from data_processing import (
    load_data,
    build_mlp_features,
    build_lstm_features,
    chronological_split,
    normalize_features,
    exclude_cols,
    prepare_dataloaders,
)
from model import WindPowerMLP, WindPowerLSTM
from train import set_seed, train_model
from evaluate import evaluate_model, compute_metrics, plot_loss_curves, plot_predictions, plot_residuals


def run_location(
    location_id: int,
    model_type: str,
    seed: int = 42,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    seq_len: int = 24,
    data_dir: str = "data",
    output_dir: str = "outputs",
) -> None:
    print(f"\n{'='*60}")
    print(f"Location {location_id} — Model: {model_type.upper()}")
    print(f"{'='*60}")

    filepath = os.path.join(data_dir, f"Location{location_id}.csv")
    df = load_data(filepath)
    print(f"  Loaded {len(df)} rows from {filepath}")

    if model_type == "mlp":
        df = build_mlp_features(df)
    else:
        df = build_lstm_features(df)

    feature_cols = [c for c in df.columns if c not in exclude_cols()]

    train_df, val_df, test_df = chronological_split(df)

    train_df, val_df, test_df, scaler = normalize_features(
        train_df, val_df, test_df, feature_cols,
    )
    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_df, val_df, test_df,
        model_type=model_type,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    sample_batch = next(iter(train_loader))[0]
    if model_type == "mlp":
        input_dim = sample_batch.shape[1]
        model = WindPowerMLP(input_dim=input_dim)
    else:
        input_dim = sample_batch.shape[2]
        model = WindPowerLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model_name = f"Location{location_id}_{model_type}"
    checkpoint_dir = os.path.join(output_dir, "trained_models")

    scaler_path = os.path.join(checkpoint_dir, f"{model_name}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        device=device,
    )

    plot_loss_curves(
        train_losses, val_losses,
        save_path=os.path.join(output_dir, "figures", f"{model_name}_loss.png"),
    )

    y_true, y_pred = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n  Test Metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}")

    plot_predictions(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "figures", f"{model_name}_predictions.png"),
    )
    plot_residuals(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "figures", f"{model_name}_residuals.png"),
    )

    results_path = os.path.join(output_dir, "predictions", f"{model_name}_test_predictions.csv")
    pd.DataFrame({"true": y_true, "predicted": y_pred}).to_csv(results_path, index=False)
    print(f"  Predictions saved to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wind Power Forecasting Pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seq-len", type=int, default=24, help="LSTM sequence length")
    parser.add_argument("--models", nargs="+", default=["mlp", "lstm"],
                        choices=["mlp", "lstm"], help="Model types to run")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs("outputs/trained_models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    for location_id in range(1, 5):
        for model_type in args.models:
            run_location(
                location_id=location_id,
                model_type=model_type,
                seed=args.seed,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                seq_len=args.seq_len,
            )

    print(f"\n{'='*60}")
    print("All runs complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
