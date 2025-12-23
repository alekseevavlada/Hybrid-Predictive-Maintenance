# pipline.py
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve,
                             roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Добавляем путь 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src import data_loader, feature_engineering, models, train

# Настройки
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path("Outputs/Metrics").mkdir(parents=True, exist_ok=True)
Path("Outputs/Models/Multiclass").mkdir(parents=True, exist_ok=True)
Path("Outputs/Models/Binary").mkdir(parents=True, exist_ok=True)
Path("Outputs/Models/RUL").mkdir(parents=True, exist_ok=True)
Path("Outputs/Plots").mkdir(parents=True, exist_ok=True)

# Загрузка конфигурации
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Использование
BATCH_SIZE = config["common"]["batch_size"]
EPOCHS_MC = config["classification"]["epochs"]

def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def save_metrics(y_true, y_pred, y_score, task, model_name, class_names=None):
    if task == "classification":
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        df = pd.DataFrame(report).T
        df.to_csv(f"Outputs/Metrics/{model_name}_classification.csv")
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pd.DataFrame({"MAE": [mae], "RMSE": [rmse]}).to_csv(f"Outputs/Metrics/{model_name}_rul.csv", index=False)

def create_rul_labels(labeled_df, horizon_hours=24):
    labeled_df = labeled_df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    labeled_df["RUL"] = np.nan
    
    for mid in labeled_df["machineID"].unique():
        sub = labeled_df[labeled_df["machineID"] == mid]
        fail_times = sub[sub["failure"] != "none"]["datetime"]
        if len(fail_times) == 0:
            continue
        fail_times = fail_times.sort_values().values
        
        for idx, row in sub.iterrows():
            current_time = row["datetime"]
            future_fails = fail_times[fail_times > current_time]
            if len(future_fails) > 0:
                rul = (future_fails[0] - current_time).total_seconds() / 3600.0
                labeled_df.loc[idx, "RUL"] = min(rul, horizon_hours)  
    
    return labeled_df.dropna(subset=["RUL"])

def main():
    # Загрузка и препроцессинг 
    telemetry, errors, maint, failures, machines = data_loader.load_datasets("Datasets")
    final_feat = feature_engineering.build_final_features(telemetry, errors, maint, machines)
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    failures = failures.sort_values(["machineID", "datetime"])

    failure_records = []

    for _, row in failures.iterrows():
        mid = row["machineID"]
        t_fail = row["datetime"]
        comp = row["failure"]
        
        window_start = t_fail - pd.Timedelta(hours=24)
        window_features = final_feat[
            (final_feat["machineID"] == mid) &
            (final_feat["datetime"] >= window_start) &
            (final_feat["datetime"] < t_fail)
        ].copy()
        window_features["failure"] = comp
        failure_records.append(window_features)

    positive_samples = pd.concat(failure_records, ignore_index=True)
    all_times = final_feat[["machineID", "datetime"]].copy()
    all_times = all_times.merge(failures, on=["machineID", "datetime"], how="left")
    all_times = all_times[all_times["failure"].isna()]  # Нет отказа именно в этот момент

    def has_failure_in_next_24h(row, failures_df):
        t = row["datetime"]
        mid = row["machineID"]
        future_fails = failures_df[
            (failures_df["machineID"] == mid) &
            (failures_df["datetime"] > t) &
            (failures_df["datetime"] <= t + pd.Timedelta(hours=24))
        ]
        return not future_fails.empty

    all_times["has_future_fail"] = all_times.apply(
        lambda r: has_failure_in_next_24h(r, failures), axis=1
    )
    negative_candidates = all_times[~all_times["has_future_fail"]]
    n_pos = len(positive_samples)
    n_neg = min(10 * n_pos, len(negative_candidates))
    negative_samples = negative_candidates.sample(n=n_neg, random_state=42).copy()
    negative_samples = negative_samples.merge(final_feat, on=["machineID", "datetime"], how="left")
    negative_samples["failure"] = "none"

    labeled_features_clean = pd.concat([positive_samples, negative_samples], ignore_index=True)
    labeled_features = labeled_features_clean.dropna(subset=["failure"])

    class_counts = labeled_features["failure"].value_counts()
    max_minority = class_counts.drop("none").max()
    undersample_none = labeled_features[
        (labeled_features["failure"] != "none") |
        (labeled_features["failure"] == "none") &
        (labeled_features.groupby("failure").cumcount() < 3.0 * max_minority)
    ]
    labeled_features = undersample_none

    split_time = labeled_features["datetime"].quantile(0.8)
    train_mask = labeled_features["datetime"] < split_time
    val_mask = ~train_mask

    non_feature_cols = ["datetime", "machineID", "failure"]
    feature_cols = [c for c in labeled_features.columns if c not in non_feature_cols]

    X_train = labeled_features.loc[train_mask, feature_cols].values.astype(np.float32)
    X_val = labeled_features.loc[val_mask, feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Multi-Class Classification
    le = LabelEncoder()
    y_train_mc = le.fit_transform(labeled_features.loc[train_mask, "failure"])
    y_val_mc = le.transform(labeled_features.loc[val_mask, "failure"])
    class_names_mc = ["none", "comp1", "comp2", "comp3", "comp4"]

    train_loader_mc = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train_mc)), batch_size=256)
    val_loader_mc = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val_mc)), batch_size=256)

    lstm_params = config["classification"]["models"]["LSTM"]
    gru_params = config["classification"]["models"]["GRU"]
    dlinear_params = config["classification"]["models"]["DLinear"]
    transformer_params = config["classification"]["models"]["Transformer"]
    rf_params = config["classification"]["models"]["RandomForest"]
    gbc_params = config["classification"]["models"]["GradientBoosting"]

    models_mc = {
        "LSTM": models.LSTMModel(X_train.shape[1], n_classes=5, hidden_size=lstm_params["hidden_size"], dropout=lstm_params["dropout"], task="classification").to(device),
        "GRU": models.GRUModel(X_train.shape[1], n_classes=5, hidden_size=gru_params["hidden_size"], dropout=gru_params["dropout"], task="classification").to(device),
        "DLinear": models.DLinear(X_train.shape[1], n_classes=5, dropout=dlinear_params["dropout"], task="classification").to(device),
        "Transformer": models.TransformerModel(X_train.shape[1], n_classes=5, hidden_size=transformer_params["hidden_size"], dropout=transformer_params["dropout"], task="classification").to(device),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=rf_params["max_depth"], class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=gbc_params["max_depth"], random_state=42)
    }

    results_mc = {}
    for name, model in models_mc.items():
        print(f"\nTraining {name}...")
        if name in ["LSTM", "GRU", "DLinear", "Transformer"]:
            model, score = train.Evaluate(model, train_loader_mc, val_loader_mc, "classification", y_train=y_train_mc, device=device)
            # Предсказание
            model.eval()
            preds = []
            with torch.no_grad():
                for X, _ in val_loader_mc:
                    preds.append(model(X.to(device)).argmax(dim=1).cpu().numpy())
            y_pred = np.concatenate(preds)
            torch.save(model.state_dict(), f"Outputs/Models/Multiclass/{name}.pth")
        else:
            model.fit(X_train, y_train_mc)
            y_pred = model.predict(X_val)
            y_score = model.predict_proba(X_val)
            score = f1_score(y_val_mc, y_pred, average='macro', zero_division=0)
            pd.to_pickle(model, f"Outputs/models/Multiclass/{name}.pkl")

        results_mc[name] = score
        save_metrics(y_val_mc, y_pred, None, "classification", name, class_names_mc)
        plot_confusion_matrix(y_val_mc, y_pred, class_names_mc, f"{name} Confusion Matrix", f"Outputs/Plots/{name}_cm.png")

    pd.DataFrame(list(results_mc.items()), columns=["Model", "Macro F1"]).to_csv("Outputs/Metrics/multiclass_summary.csv", index=False)

    # Binary Classification
    y_train_bin = (y_train_mc != 0).astype(np.int64)
    y_val_bin = (y_val_mc != 0).astype(np.int64)
    class_names_bin = ["normal", "failure"]

    train_loader_bin = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train_bin)), batch_size=256)
    val_loader_bin = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val_bin)), batch_size=256)

    models_bin = {
        "LSTM": models.LSTMModel(X_train.shape[1], n_classes=2, task="classification").to(device),
        "GRU": models.GRUModel(X_train.shape[1], n_classes=2, task="classification").to(device),
        "DLinear": models.DLinear(X_train.shape[1], n_classes=2, task="classification").to(device),
        "Transformer": models.TransformerModel(X_train.shape[1], n_classes=2, task="classification").to(device)
    }

    results_bin = {}
    for name, model in models_bin.items():
        print(f"\nTraining {name} (Binary)...")
        if name in ["LSTM", "GRU", "DLinear", "Transformer"]:
            model, score = train.Evaluate(model, train_loader_bin, val_loader_bin, "classification", y_train=y_train_bin, device=device)
            model.eval()
            preds = []
            with torch.no_grad():
                for X, _ in val_loader_bin:
                    preds.append(model(X.to(device)).argmax(dim=1).cpu().numpy())
            y_pred = np.concatenate(preds)
            torch.save(model.state_dict(), f"Outputs/Models/Binary/{name}.pth")
        else:
            model.fit(X_train, y_train_bin)
            y_pred = model.predict(X_val)
            y_score = model.predict_proba(X_val)
            score = f1_score(y_val_bin, y_pred, average='macro', zero_division=0)
            pd.to_pickle(model, f"Outputs/Models/Binary/{name}.pkl")

        results_bin[name] = score
        save_metrics(y_val_bin, y_pred, None, "classification", f"{name}_binary", class_names_bin)
        plot_confusion_matrix(y_val_bin, y_pred, class_names_bin, f"{name} (Binary) Confusion Matrix", f"Outputs/Plots/{name}_binary_cm.png")

    pd.DataFrame(list(results_bin.items()), columns=["Model", "Macro F1"]).to_csv("Outputs/Metrics/binary_summary.csv", index=False)

    # Создание RUL-метки
    labeled_rul = create_rul_labels(labeled_features, horizon_hours=24)

    # Временной сплит (сохраняем тот же сплит)
    rul_train_mask = labeled_rul["datetime"] < split_time
    rul_val_mask = ~rul_train_mask

    # Признаки и метки
    X_train_rul = labeled_rul.loc[rul_train_mask, feature_cols].values.astype(np.float32)
    y_train_rul = labeled_rul.loc[rul_train_mask, "RUL"].values.astype(np.float32)
    X_val_rul = labeled_rul.loc[rul_val_mask, feature_cols].values.astype(np.float32)
    y_val_rul = labeled_rul.loc[rul_val_mask, "RUL"].values.astype(np.float32)

    # Нормализация
    scaler_rul = StandardScaler()
    X_train_rul = scaler_rul.fit_transform(X_train_rul)
    X_val_rul = scaler_rul.transform(X_val_rul)

    # DataLoader
    train_loader_rul = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_rul), torch.FloatTensor(y_train_rul)),
        batch_size=256, shuffle=False
    )
    val_loader_rul = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_rul), torch.FloatTensor(y_val_rul)),
        batch_size=256, shuffle=False
    )

    # Модели для RUL
    models_rul = {
        "DLinear": models.DLinear(X_train_rul.shape[1], task="regression").to(device),
        "MLP": models.MLPRegressor(X_train_rul.shape[1], hidden_size=64, dropout=0.3, weight_decay=1e-4).to(device)
    }

    results_rul = {}
    for name, model in models_rul.items():
        print(f"\nTraining {name} (RUL)...")
        model, score = train.Evaluate(
            model, train_loader_rul, val_loader_rul,
            task="regression", device=device, epochs=200
        )
        mae = -score 
        
        # Предсказание
        model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in val_loader_rul:
                preds.append(model(X.to(device)).cpu().numpy())
        y_pred = np.concatenate(preds)
        
        # Метрики
        mae = mean_absolute_error(y_val_rul, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_rul, y_pred))
        results_rul[name] = {"MAE": mae, "RMSE": rmse}
        
        # Сохранение
        torch.save(model.state_dict(), f"Outputs/Models/RUL/{name}.pth")
        pd.DataFrame({"MAE": [mae], "RMSE": [rmse]}).to_csv(f"Outputs/Metrics/{name}_rul.csv", index=False)
        
        # График
        plt.figure(figsize=(10, 4))
        plt.plot(y_val_rul[:2000], label="True RUL")
        plt.plot(y_pred[:2000], label=f"Predicted RUL ({name})")
        plt.legend()
        plt.title(f"RUL Prediction for {name}")
        plt.savefig(f"Outputs/Plots/{name}_rul.png")
        plt.close()

    # Итоговая таблица
    pd.DataFrame(results_rul).T.to_csv("Outputs/Metrics/rul_summary.csv")
    print(pd.DataFrame(results_rul).T)

if __name__ == "__main__":
    main()
