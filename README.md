# Hybrid-Predictive-Maintenance

This repository implements a hybrid predictive maintenance (PdM) framework combining multi-class failure classification and Remaining Useful Life (RUL) regression. The approach is grounded in explainable AI (XAI) principles and evaluated on the Microsoft Azure PdM dataset.

All code, experiments, and results are documented in the Jupyter Notebooks provided in the `Notebooks/` directory.

## Problem Formulation

The predictive maintenance (PdM) system addresses two interrelated tasks using multivariate time-series data from 100 simulated industrial machines. Observations are aggregated at 3-hour intervals, and all features are derived solely from historical or contemporaneous data to ensure temporal validity.

**Task 1: Multi-class failure Classification**

For each machine at a given time, the model predicts whether a failure of one of four replaceable components will occur within the next 24 hours. The target is a categorical variable with five possible values: `none` (no failure) or `comp1`–`comp4` (failure of the corresponding component). 

**Task 2: Remaining Useful Life (RUL) regression**

In parallel, the system estimates the continuous time remaining until the next failure, expressed in hours and capped at 24. This provides a quantitative measure of urgency, complementing the discrete classification output. The RUL is computed as the actual time to the next recorded failure event, truncated at the 24-hour horizon to focus on the near-term operational window.

Both tasks share an identical input representation: a fixed-dimensional feature vector constructed from telemetry, error logs, maintenance records, and machine metadata.

## Feature Engineering

All features are derived from five source tables and aggregated at 3-hour intervals. The construction strictly respects temporal causality: at any time point, only past or contemporaneous data are used. The resulting feature vector comprises 52 dimensions, grouped into four categories.

**1. Temporal sensor dynamics**

- For each of the four telemetry channels (voltage, rotation, pressure, vibration):
  - Mean and standard deviation over the last 3 hours
  - Mean and standard deviation over the last 24 hours
  - Difference between 24‑h and 3‑h statistics (trend indicators)

**2. Failure precursors**

- Non-fatal error events are aggregated by type over the past 24 hours. The resulting five counts (one per error type) serve as early warning signals of abnormal operation.

**3. Maintenance history and aging**

- Days since last replacement for each of the four components
- Machine age (in years)
- Machine model type (one-hot encoded)

**4. Spectral characteristics.**

- For each telemetry channel, the two dominant frequencies and their amplitudes from a Fast Fourier Transform (FFT) applied to the last 24 raw sensor values
- FFT features are standardized independently

This representation supports both classical interpretable models (e.g., gradient-boosted trees) and modern deep learning architectures.

## Data Sources

The experiments use the synthetic Azure PdM dataset introduced in Hrnjica & Softic (2020), which simulates telemetry from 100 machines over two years. The dataset comprises:

```python
import pandas as pd

telemetry = pd.read_csv("PdM_telemetry.csv")  # Hourly sensor readings (voltage, rotation, pressure, vibration)
errors = pd.read_csv("PdM_errors.csv")        # Non-fatal error logs (5 error types)
maint = pd.read_csv("PdM_maint.csv")          # Maintenance logs (component replacements)
failures = pd.read_csv("PdM_failures.csv")    # Critical failure events (4 component types)
machines = pd.read_csv("PdM_machines.csv")    # Static metadata (model type, age)
```

> The original Azure AI Gallery link is defunct. A copy is available on Kaggle:
https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance

## Repository Structure 

```
HIBRID-PREDICTIVE-MAINTENANCE/
├── README.md
├── requirements.txt
├── Datasets/
│   ├── PdM_telemetry.csv       
│   ├── PdM_errors.csv       
│   ├── PdM_maint.csv          
│   ├── PdM_failures.csv       
│   └── PdM_machines.csv
├── Notebooks/
│   ├── Binary Classification.ipynb
│   ├── Multi-Class Classification.ipynb
│   └── RUL.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluate.py
│   └── utils.py
├── configs/
│   └── model_config.yaml
├── Models/
└── Outputs/
```

## Dependencies

```bash
# requirements.txt
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
scikit-learn==1.3.0
torch==2.0.1
pytorch-forecasting==0.10.2  
statsmodels==0.14.0
matplotlib==3.7.1
seaborn==0.12.2
tqdm==4.65.0
PyYAML==6.0  
joblib==1.3.2  
jupyter==1.0.0
ipywidgets>=7.0.0  
```

## References

1. Hrnjica, B., Softic, S. (2020). Explainable AI in Manufacturing: A Predictive Maintenance Case Study. In: Lalic, B., Majstorovic, V., Marjanovic, U., von Cieminski, G., Romero, D. (eds) Advances in Production Management Systems. Towards Smart and Digital Manufacturing. APMS 2020. IFIP Advances in Information and Communication Technology, vol 592. Springer, Cham. https://doi.org/10.1007/978-3-030-57997-5_8.
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317
3. Falcão, D., Reis, F., Farinha, J., Lavado, N., Mendes, M. Fault Detection in Industrial Equipment through Analysis of Time Series Stationarity. Algorithms 2024, 17, 455. https://doi.org/10.3390/a17100455
4. Mateus, B. C., Mendes, M., Farinha, J. T., & Martins, A. (2025). Hybrid Deep Learning for Predictive Maintenance: LSTM, GRU, CNN, and Dense Models Applied to Transformer Failure Forecasting. Energies, 18(21), 5634. https://doi.org/10.3390/en18215634
5. Azure AI Gallery. Predictive Maintenance Implementation Guide Data Sets. https://gallery.azure.ai/Experiment/Predictive-Maintenance-Implementation-Guide-Data-Sets-1
