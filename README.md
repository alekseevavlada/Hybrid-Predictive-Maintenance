# Hybrid-Predictive-Maintenance

**Early detection of industrial equipment degradation based on multivariable time series**

## Overview

This project develops a **hybrid machine learning system** for **early detection of equipment degradation** and **Remaining Useful Life (RUL) estimation** using multivariate sensor time series (e.g., temperature, pressure, vibration).

The project addresses two key objectives:

- **Мulticlass Classification** – distinguishing between five modes: normal operation (`none`) and four component failures (`comp1`–`comp4`).

- **RUL Regression** – estimating the **time until failure (in hours)**.

## Dataset

The dataset contains 5 files:

- `PdM_telemetry.csv` – which collects historical data about machine behavior (voltage, vibration, etc).

- `PdM_errors.csv` – the data about warnings and errors in the machines.

- `PdM_maint.csv` – data about replacement and maintenance for the machines.

- `PdM_failures.csv` – data when a certain machine is stopped, due to component failure.

- `PdM_machines.csv` – descriptive information about the machines.

Source: [Azure blob storage](https://gallery.azure.ai/Experiment/Predictive-Maintenance-Implementation-Guide-Data-Sets-1)

Download:[IEEEDataPort](https://ieee-dataport.org/documents/data1)

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

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

All hyperparameters are configurable in `model_config.yaml`.
