# src/feature_engineering.py
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler


def compute_fft_features(signal, n_peaks=2):
    N = len(signal)
    if N < 2:
        return np.zeros(2 * n_peaks)
    
    signal = signal - np.mean(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, d=1.0)  
    peak_indices = np.argsort(yf)[-n_peaks:][::-1]
    top_freqs = xf[peak_indices]
    top_amps = yf[peak_indices]
    result = np.concatenate([top_freqs, top_amps])

    if len(result) < 2 * n_peaks:
        result = np.pad(result, (0, 2 * n_peaks - len(result)), constant_values=0)
    elif len(result) > 2 * n_peaks:
        result = result[:2 * n_peaks]

    return result

def create_telemetry_features(telemetry):
    telemetry = telemetry.sort_values(["machineID", "datetime"])
    telemetry["datetime_3h"] = telemetry["datetime"].dt.floor("3h")
    fields = ["volt", "rotate", "pressure", "vibration"]
    agg_3h = telemetry.groupby(["machineID", "datetime_3h"])[fields].agg(["mean", "std"]).reset_index()
    agg_3h.columns = ["machineID", "datetime"] + [f"{f}{s}_3h" for f in fields for s in ["mean", "sd"]]
    telemetry_wide = telemetry.set_index("datetime")
    agg_24h_list = []
    for mid, group in telemetry_wide.groupby("machineID"):
        for f in fields:
            group[f + "mean_24h"] = group[f].rolling("24h", min_periods=1).mean()
            group[f + "sd_24h"] = group[f].rolling("24h", min_periods=1).std()
        agg_24h_list.append(group[["machineID"] + [f + "mean_24h" for f in fields] + [f + "sd_24h" for f in fields]].reset_index())
    
    agg_24h = pd.concat(agg_24h_list, ignore_index=True)
    agg_24h = agg_24h.dropna()
    feat = agg_3h.merge(agg_24h, on=["machineID", "datetime"], how="inner")

    return feat

def create_error_features(errors):
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    errors["datetime_3h"] = errors["datetime"].dt.floor("3h")
    errors["errorID"] = errors["errorID"].str.replace("error", "").astype(int)
    error_dummies = pd.get_dummies(errors, columns=["errorID"], prefix="error")
    id_cols = ["machineID", "datetime_3h"]
    error_cols = [c for c in error_dummies.columns if c.startswith("error_")]
    error_grouped = error_dummies.groupby(id_cols)[error_cols].sum().reset_index()

    temp = []
    for col in error_cols:
        pt = pd.pivot_table(error_grouped, index="datetime_3h", columns="machineID", values=col, fill_value=0)
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
    comp_rep  = comp_rep.groupby(["machineID", "datetime"]).sum().reset_index()
    comp_rep = telemetry[["datetime", "machineID"]].merge(comp_rep,
                                                      on=["datetime", "machineID"],
                                                      how="outer").fillna(0).sort_values(by=["machineID", "datetime"])
    comp_cols = ["comp1", "comp2", "comp3", "comp4"]
    for comp in comp_cols:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), "datetime"]
        comp_rep[comp] = comp_rep[comp].fillna(comp_rep["datetime"])

    comp_rep = comp_rep[comp_rep["datetime"] >= pd.to_datetime("2015-01-01")].reset_index(drop=True)
    for comp in comp_cols:
        comp_rep[comp] = (comp_rep["datetime"] - comp_rep[comp]) / np.timedelta64(1, "D")
    return comp_rep

def build_final_features(telemetry, errors, maint, machines):
    telemetry_feat = create_telemetry_features(telemetry)
    telemetry_feat = telemetry_feat.dropna()
    fields = ["volt", "rotate", "pressure", "vibration"]

    for col in fields:
        short_mean = f"{col}mean_3h"
        long_mean = f"{col}mean_24h"
        trend_col = f"{col}_trend"
        telemetry_feat[trend_col] = telemetry_feat[long_mean] - telemetry_feat[short_mean]
        short_std = f"{col}sd_3h"
        long_std = f"{col}sd_24h"
        trend_std_col = f"{col}_trend_sd"
        telemetry_feat[trend_std_col] = telemetry_feat[long_std] - telemetry_feat[short_std]

    fft_features_list = []
    window_size = 24  

    for mid in telemetry["machineID"].unique():
        sub = telemetry[telemetry["machineID"] == mid].copy()
        sub = sub.sort_values("datetime")
        for i in range(window_size, len(sub) + 1):
            window = sub.iloc[i - window_size:i]
            row = {"machineID": mid, "datetime": window["datetime"].iloc[-1]}
            for col in fields:
                signal = window[col].values
                fft_vals = compute_fft_features(signal, n_peaks=2)
                for j in range(len(fft_vals)):
                    row[f"{col}_fft{j}"] = fft_vals[j]
            fft_features_list.append(row)

    fft_df = pd.DataFrame(fft_features_list)
    telemetry_feat = telemetry_feat.merge(fft_df, on=["machineID", "datetime"], how="left")
    telemetry_feat = telemetry_feat.fillna(0)  
    fft_cols = [col for col in telemetry_feat.columns if 'fft' in col]
    scaler_fft = StandardScaler()
    telemetry_feat[fft_cols] = scaler_fft.fit_transform(telemetry_feat[fft_cols])
    error_feat = create_error_features(errors)
    comp_feat = create_maintenance_features(maint, telemetry)
    final = telemetry_feat.merge(error_feat, on=["machineID", "datetime"], how="left")
    final = final.merge(comp_feat, on=["machineID", "datetime"], how="left")
    final = final.merge(machines, on="machineID", how="left")
    final = final.fillna(0)
    final = pd.get_dummies(final, columns=["model"], prefix="model")
    return final
