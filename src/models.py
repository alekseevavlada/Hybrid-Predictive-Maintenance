# models.py
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_classes=5, dropout=0.3, task="classification"):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1 if task == "regression" else n_classes)
        self.task = task

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        out = self.dropout(h[-1])
        out = self.fc(out)
        return out.squeeze(-1) if self.task == "regression" else out

class GRUModel(LSTMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = nn.GRU(self.lstm.input_size, self.lstm.hidden_size, batch_first=True)
        del self.lstm

    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.gru(x)
        out = self.dropout(h[-1])
        out = self.fc(out)
        return out.squeeze(-1) if self.task == "regression" else out

class DLinear(nn.Module):
    def __init__(self, input_size, n_classes=5, dropout=0.3, task="classification"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        output_dim = 1 if task == "regression" else n_classes
        self.linear_trend = nn.Linear(input_size, output_dim)
        self.linear_seasonal = nn.Linear(input_size, output_dim)
        self.task = task

    def forward(self, x):
        x = self.dropout(x)
        return (self.linear_trend(x) + self.linear_seasonal(x)).squeeze(-1 if self.task == "regression" else -2)

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, n_classes=5, dropout=0.3, task="classification"):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1 if task == "regression" else n_classes)
        self.task = task

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = self.dropout(x[:, 0])
        return self.fc(x).squeeze(-1 if self.task == "regression" else -2)
    
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)