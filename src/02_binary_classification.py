# 02_binary_classification.py
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


def main():
    # Загрузка и препроцессинг 
    telemetry, errors, maint, failures, machines = data_loader.load_datasets("Datasets")
    final_feat = feature_engineering.build_final_features(telemetry, errors, maint, machines)
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    failures = failures.sort_values(["machineID", "datetime"])

    def create_binary_failure_labels(features_df, failures_df, horizons_hours=[24, 48, 168]):
        features_df = features_df.copy()
        features_df["datetime"] = pd.to_datetime(features_df["datetime"])
        failures_df = failures_df.copy()
        failures_df["datetime"] = pd.to_datetime(failures_df["datetime"])
        all_failure_times = failures_df.groupby("machineID")["datetime"].apply(list).to_dict()

        for h in horizons_hours:
            features_df[f"failure_in_{h}h"] = 0

        for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Binary labeling"):
            mid = row["machineID"]
            t_now = row["datetime"]
            if mid not in all_failure_times:
                continue
                
            future_fails = [ft for ft in all_failure_times[mid] if ft > t_now]
            if not future_fails:
                continue
                
            next_fail = min(future_fails)
            
            for h in horizons_hours:
                if (next_fail - t_now).total_seconds() <= h * 3600:
                    features_df.at[idx, f"failure_in_{h}h"] = 1
                    break  # Попал в ближайший горизонт
        
        return features_df

    labeled_features = create_binary_failure_labels(final_feat, failures, horizons_hours=[24, 48, 168])

    split_time_bin = labeled_features["datetime"].quantile(0.8)
    train_mask_bin = labeled_features["datetime"] < split_time_bin
    val_mask_bin = labeled_features["datetime"] >= split_time_bin
    feature_cols = [col for col in labeled_features.columns if not col.startswith("failure_in_") 
                    and col not in ["datetime", "machineID"]]

    horizon = 24  # Задаем горизонт отказа

    X_bin = labeled_features[feature_cols].copy()
    X_bin = pd.get_dummies(X_bin, drop_first=False)
    X_bin = X_bin.fillna(0)
    y_bin = labeled_features[f"failure_in_{horizon}h"].values

    X_train_bin = X_bin.loc[train_mask_bin].values.astype(np.float32)
    y_train_bin = y_bin[train_mask_bin].astype(np.int64)
    X_val_bin = X_bin.loc[val_mask_bin].values.astype(np.float32)
    y_val_bin = y_bin[val_mask_bin].astype(np.int64)

    scaler_bin = StandardScaler()
    X_train_scaled = scaler_bin.fit_transform(X_train_bin)
    X_val_scaled = scaler_bin.transform(X_val_bin)

    from imblearn.under_sampling import RandomUnderSampler
    undersampler = RandomUnderSampler(sampling_strategy=0.3, random_state=42) 
    X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_scaled, y_train_bin)

    X_train_t = torch.tensor(X_train_balanced, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_balanced, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_bin, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

    input_size_bin = X_train_scaled.shape[1]
    n_classes_bin = 2

    # Определяем модели и их параметры
    models_config = [
        (DLinear, {"dropout": 0.5}),
        (LSTMModel, {"hidden_size": 64, "dropout": 0.6}),
        (GRUModel, {"hidden_size": 64, "dropout": 0.6}),
        (TransformerModel, {"hidden_size": 32, "nhead": 4, "num_layers": 1, "dropout": 0.5})
    ]

    horizons = [24, 48, 168]
    binary_results = []

    for horizon in horizons:
        print(f"Evaluating for horizon: {horizon}h")
        for model_class, params in models_config:
            try:
                result = train.Evaluate(
                    model_class, params, X_train_bin, y_train_bin, X_val_bin, y_val_bin, device, horizon
                )
                binary_results.append(result)
            except Exception as e:
                print(f"Error training {model_class.__name__} for {horizon}h: {e}")
                binary_results.append({
                    "model": model_class.__name__,
                    "horizon": horizon,
                    "F2": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "AUC-PR": 0
                })

    # Бинарная классификация
    binary_df = pd.DataFrame(binary_results)
    binary_df.to_csv("/Users/admin/Documents/GitHub/Hybrid-Predictive-Maintenance/Outputs/Metrics/binary_classification_results.csv", index=False)
    
if __name__ == "__main__":
    main()
