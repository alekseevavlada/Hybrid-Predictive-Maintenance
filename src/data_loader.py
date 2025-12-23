# data_loader.py
import pandas as pd
from pathlib import Path

def load_datasets(data_dir: str):
    data_path = Path(data_dir)
    telemetry = pd.read_csv(data_path / "PdM_telemetry.csv")
    errors = pd.read_csv(data_path / "PdM_errors.csv")
    maint = pd.read_csv(data_path / "PdM_maint.csv")
    failures = pd.read_csv(data_path / "PdM_failures.csv")
    machines = pd.read_csv(data_path / "PdM_machines.csv")
    
    # Преобразуем datetime
    for df in [telemetry, errors, maint, failures]:
        df["datetime"] = pd.to_datetime(df["datetime"])
    
    return telemetry, errors, maint, failures, machines
