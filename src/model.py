import torch
import torch.nn as nn
from typing import List, Optional


class WindPowerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        if activation not in activation_map:
            raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(activation_map.keys())}")
        activation_fn = activation_map[activation]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WindPowerLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        self.lstm.reset_parameters()
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.regressor(last_out)
