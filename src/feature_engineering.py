# feature_engineering.py
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler


def create_telemetry_features(telemetry):
    telemetry = telemetry.sort_values(["machineID", "datetime"])
    telemetry["datetime_3h"] = telemetry["datetime"].dt.floor("3h")
    fields = ["volt", "rotate", "pressure", "vibration"]

    # 3h и 24h статистики
    agg_3h, agg_24h = [], []
    for col in fields:
        pt = telemetry.pivot_table(index="datetime", columns="machineID", values=col)
        agg_3h.append(pt.resample("3h").mean().unstack().rename(f"{col}mean_3h"))
        agg_3h.append(pt.resample("3h").std().unstack().rename(f"{col}sd_3h"))
        
        rolling_mean = pt.rolling(window=24, min_periods=1).mean()
        rolling_std = pt.rolling(window=24, min_periods=1).std()
        agg_24h.append(rolling_mean.resample("3h").first().unstack().rename(f"{col}mean_24h"))
        agg_24h.append(rolling_std.resample("3h").first().unstack().rename(f"{col}sd_24h"))

    feat = pd.concat(agg_3h + agg_24h, axis=1).reset_index()
    feat = feat.dropna()

    # Тренды
    for col in fields:
        feat[f"{col}_trend"] = feat[f"{col}mean_24h"] - feat[f"{col}mean_3h"]

    # FFT
    fft_list = []
    for mid in telemetry["machineID"].unique():
        sub = telemetry[telemetry["machineID"] == mid]
        for i in range(24, len(sub)):
            window = sub.iloc[i - 24:i]
            row = {"machineID": mid, "datetime": window["datetime"].iloc[-1]}
            for col in fields:
                signal = window[col].values
                fft_vals = compute_fft_features(signal, n_peaks=2)
                for j, val in enumerate(fft_vals):
                    row[f"{col}_fft{j}"] = val
            fft_list.append(row)

    fft_df = pd.DataFrame(fft_list)
    feat = feat.merge(fft_df, on=["machineID", "datetime"], how="left").fillna(0)

    # Нормализация FFT
    fft_cols = [c for c in feat.columns if "fft" in c]
    if fft_cols:
        scaler = StandardScaler()
        feat[fft_cols] = scaler.fit_transform(feat[fft_cols])

    return feat

def compute_fft_features(signal, n_peaks=2):
    if len(signal) < 2: return np.zeros(2 * n_peaks)
    signal = signal - np.mean(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal))
    idx = np.argsort(yf)[-n_peaks:][::-1]
    return np.concatenate([xf[idx], yf[idx]])

def create_error_features(errors):
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    errors["datetime_3h"] = errors["datetime"].dt.floor("3h")
    errors["errorID"] = errors["errorID"].str.replace("error", "").astype(int)
    error_dummies = pd.get_dummies(errors, columns=["errorID"], prefix="error")
    id_cols = ["machineID", "datetime_3h"]
    error_cols = [col for col in error_dummies.columns if col.startswith("error_")]
    error_grouped = error_dummies.groupby(id_cols)[error_cols].sum().reset_index()

    temp = []
    for col in error_cols:
        pt = pd.pivot_table(
            error_grouped,
            index="datetime_3h",
            columns="machineID",
            values=col,
            fill_value=0
        )
        rolling_sum = pt.rolling(window=8, min_periods=1).sum()
        resampled = rolling_sum.resample("3h", closed="left", label="right").first()
        temp.append(resampled.unstack())
      
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [f"error{i}count" for i in range(1, 6)]
    error_count = error_count.reset_index()
    error_count.rename(columns={"datetime_3h": "datetime"}, inplace=True)
    return error_count.dropna()

def create_maintenance_features(maint, telemetry):
    comp_rep = pd.get_dummies(maint.set_index("datetime")).reset_index()
    comp_rep.columns = ["datetime", "machineID", "comp1", "comp2", "comp3", "comp4"]
    comp_rep = comp_rep.groupby(["machineID", "datetime"]).sum().reset_index()

    base_telemetry = telemetry[["datetime", "machineID"]].copy()
    comp_rep = base_telemetry.merge(comp_rep, on=["datetime", "machineID"], how="outer").fillna(0)
    
    for comp in ["comp1", "comp2", "comp3", "comp4"]:
        replacement_dates = comp_rep["datetime"].where(comp_rep[comp] > 0)
        replacement_dates = replacement_dates.groupby(comp_rep["machineID"]).ffill()
        comp_rep[comp] = (
            pd.to_datetime(comp_rep["datetime"]) - 
            pd.to_datetime(replacement_dates.fillna(comp_rep["datetime"]))
        ) / pd.Timedelta(days=1)
        comp_rep[comp] = comp_rep[comp].fillna(0)
    return comp_rep


def build_final_features(telemetry, errors, maint, machines):
    tel_feat = create_telemetry_features(telemetry)
    err_feat = create_error_features(errors)
    comp_feat = create_maintenance_features(maint, telemetry)
    
    final = tel_feat.merge(err_feat, on=["machineID", "datetime"], how="left")
    final = final.merge(comp_feat, on=["machineID", "datetime"], how="left")
    final = final.merge(machines, on="machineID", how="left")
    final = final.fillna(0)
    final = pd.get_dummies(final, columns=["model"], prefix="model")
    return final
