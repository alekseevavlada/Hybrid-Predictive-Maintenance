# train.py
import copy
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (average_precision_score, fbeta_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import data_loader, feature_engineering


# Определение моделей и лосса 
class ValueMeter(object):
    """
    Вспомогательный класс, чтобы отслеживать loss и метрику
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n=1):
        self.sum += value * n
        self.total += n

    def value(self):
        return self.sum / self.total if self.total > 0 else float('nan')

def log(mode, epoch, loss_meter, f1_meter):
    """
    Логирует loss и accuracy 
    """
    print(
        f'[{mode}] Epoch: {epoch+1:02d}. '
        f'Loss: {loss_meter.value():.4f}. '
        f'F1: {100 * f1_meter.value():.2f}%'
    )

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss

class DLinear(nn.Module):
    def __init__(self, input_size, n_classes=5, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        output_dim = n_classes
        self.linear_seasonal = nn.Linear(input_size, output_dim)
        self.linear_trend = nn.Linear(input_size, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        out = self.linear_seasonal(x) + self.linear_trend(x)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, n_classes=5, nhead=4, num_layers=1, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.dropout(x)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, n_classes=5, dropout=0.6):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        h_last = self.dropout(h[-1])
        out = self.fc(h_last) 
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, n_classes=5, dropout=0.6):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.gru(x)
        h_last = self.dropout(h[-1])
        out = self.fc(h_last) 
        return out

class DLinearRegressor(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_trend = nn.Linear(input_size, 1)
        self.linear_seasonal = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.dropout(x)
        # Сумма вкладов от обеих компонент
        out = self.linear_trend(x) + self.linear_seasonal(x)
        return out.squeeze(-1)  # (B,) для регрессии

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

# Настройка устройства и директорий
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создаем все необходимые директории
for d in ["Outputs/Metrics", "Outputs/Models/Binary", "Outputs/Plots"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Обновленная функция Evaluate 
def Evaluate(model_class, model_params, train_loader, val_loader, input_size, device, horizon, epochs=200):
    model = model_class(input_size=input_size, n_classes=2, **model_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    best_f2 = 0
    patience = 10
    counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Валидация
        model.eval()
        y_true, y_proba = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                prob = torch.softmax(out, dim=1)[:, 1]
                y_true.append(y.cpu().numpy())
                y_proba.append(prob.cpu().numpy())
                
        y_true = np.concatenate(y_true)
        y_proba = np.concatenate(y_proba)
        f2 = fbeta_score(y_true, y_proba > 0.5, beta=2, zero_division=0)
        
        if f2 > best_f2:
            best_f2 = f2
            counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Подбор порога
    thresholds = np.linspace(0.1, 0.9, 50)
    f2_scores = [fbeta_score(y_true, y_proba > t, beta=2, zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f2_scores)]
    y_pred = (y_proba > best_t).astype(int)

    model_path = f"Outputs/Models/Binary/{model_class.__name__}_horizon{horizon}h.pth"
    torch.save(best_state, model_path)
    
    return {
        "model": model_class.__name__,
        "horizon": horizon,
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "AUC-PR": average_precision_score(y_true, y_proba)
    }


def Evaluate_RUL(model_name, X_train, y_train, X_val, y_val, params, input_size, device, epochs=50):
    # Нормализация
    scaler_rul = StandardScaler()
    X_train_scaled = scaler_rul.fit_transform(X_train)
    X_val_scaled = scaler_rul.transform(X_val)

    X_tr = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=256, shuffle=False)
    
    if model_name == "DLinear":
        model = DLinearRegressor(input_size, dropout=params["dropout"]).to(device)
    else:
        model = MLPRegressor(input_size, hidden_size=params["hidden_size"], dropout=params["dropout"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )
    criterion = nn.MSELoss()
    patience = 10
    counter = 0
    
    best_val_mae = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    # История обучения
    train_mae_history, val_mae_history = [], []
    train_loss_history, val_loss_history = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss_meter = ValueMeter()
        train_mae_meter = ValueMeter()
        for X, y in tqdm(train_loader, desc=f"Train RUL {epoch+1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_meter.add(loss.item(), X.size(0))
            mae = torch.mean(torch.abs(out - y)).item()
            train_mae_meter.add(mae, X.size(0))
        
        train_mae_history.append(train_mae_meter.value())
        train_loss_history.append(train_loss_meter.value())
        
        # Валидация
        model.eval()
        val_loss_meter = ValueMeter()
        val_mae_meter = ValueMeter()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss_meter.add(loss.item(), X.size(0))
                mae = torch.mean(torch.abs(out - y)).item()
                val_mae_meter.add(mae, X.size(0))
        
        val_mae = val_mae_meter.value()
        val_mae_history.append(val_mae)
        val_loss_history.append(val_loss_meter.value())
        
        # print(f"[RUL] Epoch {epoch+1}. Train MAE: {train_mae_meter.value():.2f}, Val MAE: {val_mae:.2f}")
        
        # Early Stopping на основе MAE 
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                # print(f"Early stopping on epoch {epoch+1}")
                break
                
    model.load_state_dict(best_model_state)
    
    # Финальное предсказание на валидации
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            out = model(X)
            y_pred_list.append(out.cpu().numpy())
    y_pred = np.concatenate(y_pred_list, axis=0)

    history = {
        'train_mae': train_mae_history,
        'val_mae': val_mae_history,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'best_val_mae': best_val_mae
    }
    
    return y_pred, history, best_val_mae, model
