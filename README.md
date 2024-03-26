# DR_TelegramBot

## ML/DL model

Python environment is managed by `conda`. To create the environment, run:
```bash
conda env create -f environment.yml
```

Data (DPR) updated to 2024-03-25 Period 33.

1. Data analysis: `data_analysis.ipynb`
2. Data processing demo: `net_demand.ipynb`
3. Model training: 
   - `lstm_simple.ipynb`
   - `lstm_bi.ipynb`
4. Forecasting: `forecast.ipynb` -> `src/forecast.py`

Scalers and models are saved under `model/` directory, using datetime as subfolder name.