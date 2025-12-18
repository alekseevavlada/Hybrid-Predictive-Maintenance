# Hybrid-Predictive-Maintenance

This project implements a **hybrid predictive maintenance (PdM) framework** that integrates **multi-class failure classification** and **Remaining Useful Life (RUL) regression** for industrial machinery. This repository demonstrates a pipeline built on **Explainable AI (XAI)** principles and leverages both classical machine learning and modern deep learning architectures for time-series sensor data from manufacturing environments.

> Note: The complete implementation, including data loading, feature engineering, model definition, training loops, and evaluation metrics, is provided in the accompanying Jupyter Notebooks. 

## Overview

Predictive maintenance (PdM) represents a paradigm shift from traditional reactive or scheduled maintenance toward data-driven, proactive asset management. By leveraging operational telemetry and event logs, PdM systems aim to anticipate incipient failures, thereby minimizing unplanned downtime, optimizing maintenance expenditures, and extending the service life of critical machinery. 

The foundation of this project is a **causally consistent, high-dimensional feature representation** derived from heterogeneous industrial data streams, including sensor telemetry, error logs, maintenance records, and static machine metadata.

The principal methodological point lies in the **shared feature space architecture**: a single, temporally aligned feature matrix, constructed at a 3-hour resolution, serves as input to both the classification and regression modules. This design enforces semantic and temporal coherence between the two predictive objectives and establishes a natural pathway toward future integration into a multi-task learning (MTL) framework, where shared representations can be optimized end-to-end for joint performance.

The feature engineering pipeline systematically encodes four orthogonal yet complementary facets of machine health degradation:

- **Temporal sensor dynamics**: Rolling statistical descriptors (mean and standard deviation) are computed over short-term (3-hour) and long-term (24-hour) windows for each sensor channel (voltage, rotation, pressure, vibration). 
- **Failure precursors**: Non-fatal error events are aggregated into cumulative counts over a 24-hour sliding window for each of five error types.
- **Maintenance history and aging**: For each of four replaceable components, the time elapsed since the last replacement is computed, alongside machine age (in years). 
- **Spectral characteristics**: Fast Fourier Transform (FFT) is applied to the most recent 24 raw telemetry observations per sensor to extract dominant frequency components and their amplitudes. 
  
Critically, all features are engineered under a **strict temporal causality constraint**: at any given timestamp, only historical or contemporaneous data are used. No future information is permitted to influence feature computation. 

## Data Sources

Based on the open-source Azure PdM dataset used in [Hrnjica & Softic (2020)](https://link.springer.com/chapter/10.1007/978-3-030-57997-5_8), which simulates telemetry from 100 machines over two years. The dataset includes:

```python
import pandas as pd

telemetry = pd.read_csv("PdM_telemetry.csv")  # Hourly sensor readings (voltage, rotation, pressure, vibration)
errors = pd.read_csv("PdM_errors.csv")        # Non-fatal error logs (5 error types)
maint = pd.read_csv("PdM_maint.csv")          # Maintenance logs (component replacements)
failures = pd.read_csv("PdM_failures.csv")    # Critical failure events (4 component types)
machines = pd.read_csv("PdM_machines.csv")    # Static metadata (model type, age)
```

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
│   ├── Classification.ipynb
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
├── Outputs/
├── Report.pdf
└── Presentation.pdf
```

## Dependencies

```bash
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
torch>=1.10
matplotlib
seaborn
tqdm
statsmodels
```

## References

1. Hrnjica, B., Softic, S. (2020). Explainable AI in Manufacturing: A Predictive Maintenance Case Study. In: Lalic, B., Majstorovic, V., Marjanovic, U., von Cieminski, G., Romero, D. (eds) Advances in Production Management Systems. Towards Smart and Digital Manufacturing. APMS 2020. IFIP Advances in Information and Communication Technology, vol 592. Springer, Cham. https://doi.org/10.1007/978-3-030-57997-5_8.
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317
3. Azure AI Gallery. Predictive Maintenance Implementation Guide Data Sets. https://gallery.azure.ai/Experiment/Predictive-Maintenance-Implementation-Guide-Data-Sets-1
