# 03_rul_regression.py
import os
import sys
import warnings

import joblib

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, fbeta_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

matplotlib.use('Agg')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import data_loader, feature_engineering, models, train
from src.models import (DLinear, DLinearRegressor, GRUModel, LSTMModel,
                        MLPRegressor, TransformerModel)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in ["Outputs/Metrics", "Outputs/Models/Multiclass", "Outputs/Models/RUL", "Outputs/Plots"]:
    Path(d).mkdir(parents=True, exist_ok=True)

with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)


def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def save_classification_report(y_true, y_pred, target_names, path):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    pd.DataFrame(report).T.to_csv(path)


def walk_forward_splits(df, n_splits=5, min_train_ratio=0.3):
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    min_train_size = int(n * min_train_ratio)
    step = (n - min_train_size) // n_splits
    splits = []
    for i in range(n_splits):
        train_end = min_train_size + i * step
        if train_end >= n: break
        val_end = min(train_end + step, n)
        train_idx = df.index[:train_end]
        val_idx = df.index[train_end:val_end]
        splits.append((train_idx, val_idx))
    return splits


def main():
    # Загрузка и препроцессинг 
    telemetry, errors, maint, failures, machines = data_loader.load_datasets("Datasets")
    final_feat = feature_engineering.build_final_features(telemetry, errors, maint, machines)

    def calculate_rul(features_df, failures_df, max_rul=24):
        features_df = features_df.copy()
        features_df["datetime"] = pd.to_datetime(features_df["datetime"])
        failures_df = failures_df.copy()
        failures_df["datetime"] = pd.to_datetime(failures_df["datetime"])
        
        rul_list = []
        failures_grouped = failures_df.groupby("machineID")["datetime"].apply(list).to_dict()
        
        for _, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Calculating RUL"):
            machine_id = row["machineID"]
            current_time = row["datetime"]
            machine_failures = failures_grouped.get(machine_id, [])
            future_failures = [ft for ft in machine_failures if ft > current_time]
            
            if future_failures:
                next_failure = min(future_failures)
                rul_hours = (next_failure - current_time).total_seconds() / 3600.0
                rul_hours = min(rul_hours, max_rul)
            else:
                rul_hours = max_rul
            rul_list.append(rul_hours)
        return pd.Series(rul_list, index=features_df.index)

    rul_series = calculate_rul(final_feat, failures, max_rul=24)
    rul_features = final_feat.copy()
    rul_features["RUL"] = rul_series.values
    rul_features = rul_features.dropna(subset=["RUL"])
    rul_features = rul_features.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    splits = walk_forward_splits(rul_features, n_splits=5, min_train_ratio=0.4)

    from itertools import product
    feature_columns = [col for col in rul_features.columns if col not in ["datetime", "machineID", "RUL"]]
    X_full_df = rul_features[feature_columns].copy() 
    X_full_df = pd.get_dummies(X_full_df, drop_first=False)
    X_full_df = X_full_df.fillna(0)

    X_full = rul_features[feature_columns].values
    y_full = rul_features["RUL"].values
    input_size = X_full.shape[1]

    param_grid = {
        "model": ["DLinear", "MLP"],
        "lr": [1e-3, 1e-4],
        "dropout": [0.3],
        "hidden_size": [64],  # Игнорируется для DLinear
        "weight_decay": [1e-4]
    }

    def dict_product(d):
        keys = d.keys()
        for vals in product(*d.values()):
            yield dict(zip(keys, vals))

    results = []
    for config in tqdm(list(dict_product(param_grid)), desc="Tuning configs"):
        model_name = config["model"]
        filtered_params = {
            k: v for k, v in config.items()
            if not (k == "hidden_size" and model_name == "DLinear")
        }
        fold_scores = []
        for train_idx, val_idx in splits:
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            y_train, y_val = y_full[train_idx], y_full[val_idx]
            try:
                _, _, val_mae, model = train.Evaluate_RUL(
                    model_name=config["model"],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    params=filtered_params,
                    input_size=input_size,
                    device=device,
                    epochs=50
                )
                fold_scores.append(val_mae)
            except Exception as e:
                print(f"Error in fold: {e}")
                fold_scores.append(float('inf'))
                
        results.append({**config, "cv_mae": np.mean(fold_scores)})

    results_df = pd.DataFrame(results)
    best_config = results_df.loc[results_df["cv_mae"].idxmin()]

    split_time_final = rul_features["datetime"].quantile(0.8)
    train_mask = rul_features["datetime"] < split_time_final
    val_mask = rul_features["datetime"] >= split_time_final

    X_train_final = rul_features.loc[train_mask, feature_columns].values
    y_train_final = rul_features.loc[train_mask, "RUL"].values
    X_val_final = rul_features.loc[val_mask, feature_columns].values
    y_val_final = rul_features.loc[val_mask, "RUL"].values

    # Обучение финальной модели
    y_pred_final, final_history, _, final_rul_model = train.Evaluate_RUL(
        model_name=best_config["model"],
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        params=best_config.to_dict(),
        input_size=input_size,
        device=device,
        epochs=100
    )

    mae = mean_absolute_error(y_val_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_val_final, y_pred_final))
    r2 = r2_score(y_val_final, y_pred_final)
    mape = np.mean(np.abs((y_val_final - y_pred_final) / (y_val_final + 1e-8))) * 100  # Avoid div by zero

    # print(f"Final {best_config['model']} RUL Metrics:")
    # print(f"MAE: {mae:.2f} hours")
    # print(f"RMSE: {rmse:.2f} hours")
    # print(f"R^2: {r2:.4f}")
    # print(f"MAPE (%): {mape:.4f}")

    # Сохранение графиков
    plots_dir = Path("/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Визуализация: График истинного RUL vs Предсказанного (для первых 2000 точек)
    plt.figure(figsize=(12, 4))

    plt.plot(y_val_final[:2000], label='True RUL', alpha=0.7)
    plt.plot(y_pred_final[:2000], label=f'{best_config["model"]} RUL', alpha=0.7)
    plt.title(f'{best_config["model"]}: True vs Predicted RUL (first 2000 samples)')
    plt.xlabel('Sample')
    plt.ylabel('RUL (hours)')
    plt.legend()
    plt.grid(True)
    # Сохраняем и закрываем
    plt.savefig(plots_dir / "rul_true_vs_pred.png", bbox_inches='tight', dpi=150)
    plt.close()

    # Визуализация: График обучения
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(final_history['train_mae'], label='Train MAE')
    plt.plot(final_history['val_mae'], label='Val MAE')
    plt.title('Training MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(final_history['train_loss'], label='Train Loss (MSE)')
    plt.plot(final_history['val_loss'], label='Val Loss (MSE)')
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # Сохраняем и закрываем
    plt.savefig(plots_dir / "rul_training_history.png", bbox_inches='tight', dpi=150)
    plt.close()

    # Базовый DLinear (без настройки)
    y_pred_dlinear_base, _, _, _ = train.Evaluate_RUL(
        model_name="DLinear",
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        params={"dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
        input_size=input_size,
        device=device,
        epochs=100
    )

    # Базовый MLP
    y_pred_mlp_base, _, _, _ = train.Evaluate_RUL(
        model_name="MLP",
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        params={"dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "hidden_size": 128},
        input_size=input_size,
        device=device,
        epochs=100
    )

    models_preds = {
        "DLinear (default)": y_pred_dlinear_base,
        "MLP (default)": y_pred_mlp_base,
        f"Best ({best_config['model']})": y_pred_final
    }

    def compute_all_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Avoid div by zero
        return {"MAE": mae, "RMSE": rmse, "R^2": r2, "MAPE (%)": mape}

    comparison = {}
    for name, y_pred in models_preds.items():
        metrics = compute_all_metrics(y_val_final, y_pred)
        comparison[name] = metrics

    comparison_df = pd.DataFrame(comparison).T
    comparison_df.to_csv("/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Metrics/rul_regression_comparison_df.csv", index=False)
    
if __name__ == "__main__":
    main()
