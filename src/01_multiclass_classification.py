# 01_multiclass_classification.py
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

from src import data_loader, feature_engineering, models
from src.models import (DLinear, GRUModel, LSTMModel, MLPRegressor,
                        TransformerModel)

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


def main():
    # Загрузка и препроцессинг 
    telemetry, errors, maint, failures, machines = data_loader.load_datasets("Datasets")
    final_feat = feature_engineering.build_final_features(telemetry, errors, maint, machines)
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    failures = failures.sort_values(["machineID", "datetime"])
    failure_records = []

    # Многоклассовая разметка 
    for _, row in failures.iterrows():
        mid, t_fail, comp = row["machineID"], row["datetime"], row["failure"]
        window_start = t_fail - pd.Timedelta(hours=24)
        window_features = final_feat[
            (final_feat["machineID"] == mid) &
            (final_feat["datetime"] >= window_start) &
            (final_feat["datetime"] < t_fail)
        ].copy()

        window_features["failure"] = comp
        failure_records.append(window_features)

    positive_samples = pd.concat(failure_records, ignore_index=True)

    # Отрицательные примеры
    all_times = final_feat[["machineID", "datetime"]].copy()
    all_times = all_times.merge(failures, on=["machineID", "datetime"], how="left")
    all_times = all_times[all_times["failure"].isna()]

    def has_future_fail(row, failures_df):
        t, mid = row["datetime"], row["machineID"]
        future_fails = failures_df[
            (failures_df["machineID"] == mid) &
            (failures_df["datetime"] > t) &
            (failures_df["datetime"] <= t + pd.Timedelta(hours=24))
        ]
        return not future_fails.empty

    all_times["has_future_fail"] = all_times.apply(
        lambda r: has_future_fail(r, failures), axis=1
    )
    negative_candidates = all_times[~all_times["has_future_fail"]]
    n_pos = len(positive_samples)
    n_neg = min(10 * n_pos, len(negative_candidates))
    negative_samples = negative_candidates.sample(n=n_neg, random_state=42).copy()
    negative_samples = negative_samples.merge(final_feat, on=["machineID", "datetime"], how="left")
    negative_samples["failure"] = "none"

    labeled_features_clean = pd.concat([positive_samples, negative_samples], ignore_index=True)
    labeled_features = labeled_features_clean.dropna(subset=["failure"])

    # Undersampling
    class_counts = labeled_features["failure"].value_counts()
    max_minority = class_counts.drop("none").max()
    undersample_none = labeled_features[
        (labeled_features["failure"] != "none") |
        (labeled_features["failure"] == "none") &
        (labeled_features.groupby("failure").cumcount() < 3.0 * max_minority)
    ]
    labeled_features = undersample_none

    # Сплит
    split_time = labeled_features["datetime"].quantile(0.8)
    train_mask = labeled_features["datetime"] < split_time
    val_mask = labeled_features["datetime"] >= split_time

    feature_cols = [c for c in labeled_features.columns if c not in ["datetime", "machineID", "failure"]]
    X = labeled_features.drop(columns=["datetime", "machineID", "failure"])
    y = labeled_features["failure"]

    label_map = {"none": 0, "comp1": 1, "comp2": 2, "comp3": 3, "comp4": 4}
    y = labeled_features["failure"]
    y_encoded = y.map(label_map).astype(np.int64)
    X = pd.get_dummies(X, drop_first=False)  
    input_size = X.shape[1]

    X_train = X.loc[train_mask].values.astype(np.float32)
    y_train = y_encoded[train_mask].values.astype(np.int64)
    X_val = X.loc[val_mask].values.astype(np.float32)
    y_val = y_encoded[val_mask].values.astype(np.int64)

    # Multiclass Classification
    scaler_mc = joblib.load("/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/scaler_clf.pkl")
    X_val_scaled = scaler_mc.transform(X_val)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    def load_torch_model(model_class, path, **kwargs):
        model = model_class(**kwargs).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    models_mc = {
        "LSTM": load_torch_model(
            LSTMModel,
            "/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/lstm_model.pth",
            input_size=input_size, hidden_size=128, n_classes=5, dropout=0.9
        ),
        "GRU": load_torch_model(
            GRUModel,
            "/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/gru_model.pth",
            input_size=input_size, hidden_size=128, n_classes=5, dropout=0.9
        ),
        "DLinear": load_torch_model(
            DLinear,
            "/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/dlinear_model.pth",
            input_size=input_size, n_classes=5, dropout=0.2
        ),
        "Transformer": load_torch_model(
            TransformerModel,
            "/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/transformer_model.pth",
            input_size=input_size, hidden_size=64, n_classes=5, dropout=0.5
        ),
        "RandomForest": joblib.load("/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/rf_balanced.pth"),
        "GradientBoosting": joblib.load("/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Models/Multiclass/gbc_balanced.pth")
    }

    # Оценка моделей (без обучения)
    results_mc = {}
    target_names = ["none", "comp1", "comp2", "comp3", "comp4"]

    for name, model in models_mc.items():
        print(f"Evaluating {name} (Multiclass)...")
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                logits = model(X_val_tensor)
                y_score = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            y_score = model.predict_proba(X_val_scaled)

        best_thresholds = [0.5]  
        for class_id in [1, 2, 3, 4]:
            f2_scores = [
                fbeta_score(
                    (y_val == class_id).astype(int),
                    (y_score[:, class_id] > t).astype(int),
                    beta=2, zero_division=0
                )
                for t in np.linspace(0.1, 0.9, 100)
            ]
            best_t = np.linspace(0.1, 0.9, 100)[np.argmax(f2_scores)]
            best_thresholds.append(best_t)
        adjusted_scores = y_score / np.array(best_thresholds)
        y_pred = np.argmax(adjusted_scores, axis=1)
    
        results_mc[name] = f1_score(y_val, y_pred, average="macro")
        save_classification_report(y_val, y_pred, target_names, f"/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Metrics/{name}_multiclass.csv")
        plot_confusion_matrix(y_val, y_pred, target_names, f"{name} CM", f"/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Plots/{name}_cm.png")

    pd.DataFrame(list(results_mc.items()), columns=["Model", "Macro F1"]).to_csv(
        "/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Metrics/multiclass_classification_loaded.csv", index=False
    )
    
if __name__ == "__main__":
    main()
