import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List


RAW_FEATURES = [
    "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
    "windspeed_10m", "windspeed_100m", "windgusts_10m",
]

WIND_DIR_COLS = ["winddirection_10m", "winddirection_100m"]

TARGET_COL = "Power"

LAGS = [1, 2, 3, 6, 12, 24]

ROLLING_WINDOWS = [3, 6, 24]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["Time"].dt.hour
    df["dayofweek"] = df["Time"].dt.dayofweek
    df["month"] = df["Time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def encode_circular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in WIND_DIR_COLS:
        rad = np.deg2rad(df[col])
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
    return df


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w).std()
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


class WindPowerMLPDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        lag_cols = [c for c in df.columns if "_lag_" in c]
        roll_cols = [c for c in df.columns if "_roll_" in c]
        temporal_cols = [
            "hour_sin", "hour_cos", "dayofweek", "month_sin", "month_cos",
        ]
        wind_dir_cols_enc = [
            "winddirection_10m_sin", "winddirection_10m_cos",
            "winddirection_100m_sin", "winddirection_100m_cos",
        ]
        feature_cols = RAW_FEATURES + wind_dir_cols_enc + temporal_cols + lag_cols + roll_cols
        feature_cols = [c for c in feature_cols if c in df.columns]
        data = df[feature_cols].values.astype(np.float32)
        targets = df[TARGET_COL].values.astype(np.float32)
        valid = ~np.isnan(data).any(axis=1) & ~np.isnan(targets)
        self.X = torch.from_numpy(data[valid])
        self.y = torch.from_numpy(targets[valid]).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class WindPowerLSTMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 24):
        temporal_cols = [
            "hour_sin", "hour_cos", "dayofweek", "month_sin", "month_cos",
        ]
        wind_dir_cols_enc = [
            "winddirection_10m_sin", "winddirection_10m_cos",
            "winddirection_100m_sin", "winddirection_100m_cos",
        ]
        feature_cols = RAW_FEATURES + wind_dir_cols_enc + temporal_cols
        feature_cols = [c for c in feature_cols if c in df.columns]
        values = df[feature_cols].values.astype(np.float32)
        targets = df[TARGET_COL].values.astype(np.float32)
        self.seq_len = seq_len
        self.X, self.y = self._create_sequences(values, targets)

    def _create_sequences(
        self,
        values: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(values)):
            X_seq.append(values[i - self.seq_len : i])
            y_seq.append(targets[i])
        return torch.from_numpy(np.array(X_seq)), torch.from_numpy(np.array(y_seq)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, val_df, test_df, scaler


def prepare_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = "mlp",
    batch_size: int = 64,
    seq_len: int = 24,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if model_type == "mlp":
        train_ds = WindPowerMLPDataset(train_df)
        val_ds = WindPowerMLPDataset(val_df)
        test_ds = WindPowerMLPDataset(test_df)
    elif model_type == "lstm":
        train_ds = WindPowerLSTMDataset(train_df, seq_len=seq_len)
        val_ds = WindPowerLSTMDataset(val_df, seq_len=seq_len)
        test_ds = WindPowerLSTMDataset(test_df, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_mlp_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = encode_circular(df)
    df = create_temporal_features(df)
    df = create_lag_features(df, ["windspeed_10m", "windspeed_100m", "windgusts_10m", TARGET_COL], LAGS)
    df = create_rolling_features(df, ["windspeed_10m", "windspeed_100m", "windgusts_10m"], ROLLING_WINDOWS)
    df = df.dropna().reset_index(drop=True)
    return df


def build_lstm_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = encode_circular(df)
    df = create_temporal_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


def get_normalization_cols(model_type: str) -> List[str]:
    base_cols = ["temperature_2m", "relativehumidity_2m", "dewpoint_2m",
                 "windspeed_10m", "windspeed_100m", "windgusts_10m",
                 "winddirection_10m_sin", "winddirection_10m_cos",
                 "winddirection_100m_sin", "winddirection_100m_cos",
                 "hour_sin", "hour_cos", "dayofweek", "month_sin", "month_cos"]
    if model_type == "mlp":
        lag_cols = [f"{v}_lag_{l}" for v in ["windspeed_10m", "windspeed_100m", "windgusts_10m", TARGET_COL] for l in LAGS]
        roll_cols = [f"{v}_roll_{s}_{w}" for v in ["windspeed_10m", "windspeed_100m", "windgusts_10m"] for w in ROLLING_WINDOWS for s in ["mean", "std"]]
        return base_cols + lag_cols + roll_cols
    return base_cols
