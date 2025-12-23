# train.py
import copy

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Загрузка конфигурации
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

class ValueMeter(object):
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
    
def Evaluate(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str = "classification",
    epochs: int = 100,
    y_train: np.ndarray = None,
    device: str = "cpu",
    return_predictions: bool = False,
    return_history: bool = False
):
    model = model.to(device)

    # --- Определение функции потерь ---
    if task == "classification":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.HuberLoss(delta=1.0)  # Устойчив к выбросам

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["classification"]["lr"]) if task == "classification" else float(config["regression"]["lr"]),
        weight_decay=float(config["classification"]["weight_decay"]) if task == "classification" else float(config["regression"]["weight_decay"])
    )
    
    # Инициализация early stopping
    best_score = -float('inf') if task == "classification" else float('inf')
    patience, counter = 15, 0
    best_state = copy.deepcopy(model.state_dict())
    
    # История обучения 
    if return_history:
        history = {"train_loss": [], "val_loss": [], "train_score": [], "val_score": []}

    for epoch in range(epochs):
        # Обучение 
        model.train()
        train_loss_meter = 0.0
        all_preds_train, all_trues_train = [], []

        for X, y in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_meter += loss.item() * X.size(0)

            if task == "classification":
                all_preds_train.append(out.argmax(dim=1).cpu().numpy())
                all_trues_train.append(y.cpu().numpy())

        train_loss = train_loss_meter / len(train_loader.dataset)

        # Метрика на трейне
        if task == "classification":
            train_preds = np.concatenate(all_preds_train)
            train_trues = np.concatenate(all_trues_train)
            train_score = f1_score(train_trues, train_preds, average='macro')
        else:
            train_score = 0.0

        # Валидация 
        model.eval()
        val_loss_meter = 0.0
        all_preds_val, all_trues_val = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss_meter += loss.item() * X.size(0)
                all_preds_val.append(out.cpu().numpy())
                all_trues_val.append(y.cpu().numpy())

        val_loss = val_loss_meter / len(val_loader.dataset)
        val_preds = np.concatenate(all_preds_val)
        val_trues = np.concatenate(all_trues_val)

        if task == "classification":
            val_score = f1_score(val_trues, val_preds.argmax(axis=1), average='macro')
        else:
            val_score = -mean_absolute_error(val_trues, val_preds)  # Максимизируем MAE

        # Early Stopping 
        is_better = val_score > best_score if task == "classification" else val_score > best_score
        if is_better:
            best_score = val_score
            counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                break

        if return_history:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_score"].append(train_score)
            history["val_score"].append(val_score)

    model.load_state_dict(best_state)

    result = [model, best_score]
    if return_predictions:
        result.append(val_preds)
    if return_history:
        result.append(history)
    
    return tuple(result) if len(result) > 2 else (model, best_score)