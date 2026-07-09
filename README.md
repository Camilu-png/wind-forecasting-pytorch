# Wind Power Forecasting

PyTorch-based wind turbine power output forecasting using meteorological data from 4 locations.

## Dataset

Hourly meteorological readings (2017–2021) from [Kaggle](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting). Each location has 43,800 hourly samples.

**Features**: temperature, humidity, dewpoint, wind speed (10m & 100m), wind direction (10m & 100m), wind gusts.

**Target**: normalized turbine power output (0–1).

## Project Structure

```
├── data/               # Raw CSV files (Location1–4.csv)
├── outputs/
│   ├── trained_models/ # Model checkpoints (.pt)
│   ├── figures/        # Loss curves, predictions, residuals
│   └── predictions/    # Test set predictions (CSV)
├── src/
│   ├── data_processing.py  # Load, featurize, split, normalize, datasets
│   ├── model.py            # WindPowerMLP and WindPowerLSTM
│   ├── train.py            # Training loop, early stopping, checkpointing
│   ├── evaluate.py         # Metrics and visualization
│   └── main.py             # Pipeline orchestrator
├── requirements.txt
└── README.md
```

## Preprocessing Pipeline

1. **Load & sort** — parse datetime, ensure chronological order
2. **Temporal features** — hour (sin/cos), dayofweek, month (sin/cos)
3. **Wind direction encoding** — sin/cos transform to handle 0/360 wraparound
4. **MLP features only** — lag features (t-1, t-2, t-3, t-6, t-12, t-24) for wind speeds, gusts, and power; rolling mean/std over 3h, 6h, 24h windows
5. **Chronological split** — 70% train / 15% validation / 15% test (no shuffle)
6. **Normalization** — StandardScaler fitted on training set only, applied to val/test

## Models

### WindPowerMLP
Feedforward network with configurable hidden layers, activation, and dropout. Xavier initialization. Takes a flat feature vector with engineered lags and rolling statistics.

### WindPowerLSTM
2-layer LSTM that processes 24-hour sequences of raw meteorological features. Orthogonal initialization for recurrent weights, Xavier for linear layers. Uses last hidden state for prediction.

## Training

- **Loss**: MSE
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: patience=10 epochs
- **Checkpointing**: saves model with lowest validation loss
- **Seed**: 42 (deterministic)

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

## Usage

```bash
# Train MLP on all locations
python src/main.py --models mlp

# Train both models
python src/main.py --models mlp lstm

# Custom hyperparameters
python src/main.py --models mlp --batch-size 128 --lr 5e-4 --epochs 150
```

## Results

After running, outputs are organized by location and model:

```
outputs/
├── trained_models/
│   ├── Location1_mlp.pt
│   ├── Location1_lstm.pt
│   └── ...
├── figures/
│   ├── Location1_mlp_loss.png
│   ├── Location1_mlp_predictions.png
│   ├── Location1_mlp_residuals.png
│   └── ...
└── predictions/
    ├── Location1_mlp_test_predictions.csv
    └── ...
```

## Reproducibility

- Fixed random seed (42)
- Deterministic cuDNN operations
- Chronological split (no shuffling)
- Normalization parameters saved per location
- Full hyperparameter configuration in source
