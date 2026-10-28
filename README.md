# Retail Demand Forecasting: SARIMA vs Prophet vs LightGBM

Forecasting 28 days of store × department demand on the Walmart **M5** dataset, with a **PySpark** feature
pipeline and a rolling-origin backtest of four models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaishvin/m5-demand-forecasting/blob/main/notebooks/m5_demand_forecasting.ipynb)

## Business problem
Demand planners need reliable short-term forecasts to set replenishment and inventory levels. Classic
statistical models (ARIMA/SARIMA) handle seasonality well but struggle with promotions, price changes and
events. This project measures how much accuracy ML methods add on real retail data, and which drivers matter.

## Approach
| Step | What | Tools |
|---|---|---|
| 1. Ingest | Reshape 1,941 days of wide SKU-level sales to long format; join calendar and weekly prices | PySpark |
| 2. Features | Discount depth vs 8-week reference price, price index, SNAP days, events, lags (28–364 days), rolling means/std | PySpark window functions |
| 3. Models | Seasonal naive (baseline), SARIMA(1,0,1)(1,1,1,7), Prophet with events + price regressors, global LightGBM (Tweedie loss) | statsmodels, Prophet, LightGBM |
| 4. Backtest | Rolling-origin CV: 3 folds × 28-day horizon | pandas |
| 5. Evaluate | WMAPE, forecast accuracy (1 − WMAPE), bias; by department and event vs normal days | pandas, matplotlib |
| 6. Forecast | Retrain best model on all history and forecast the next 28 days | — |

All lag features are ≥ 28 days, so one LightGBM model forecasts the full 28-day horizon directly, with no
recursive forecasting and no leakage.

## Results
<!-- Fill in after running the notebook on the real data -->
| Model | WMAPE | Forecast accuracy | vs baseline |
|---|---|---|---|
| Seasonal naive | __% | __% | — |
| SARIMA | __% | __% | __% |
| Prophet | __% | __% | __% |
| LightGBM | __% | __% | __% |

![Model comparison](results/model_comparison.png)
![Forecast example](results/forecast_example.png)
![Feature importance](results/feature_importance.png)

**Key findings**
- _Best model and accuracy gain over baseline_
- _Top demand drivers from feature importance_
- _Where the models struggle (event days, low-volume departments)_

## Run it
1. Click **Open in Colab** above.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets and accept the
   [M5 competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules).
3. Runtime → Run all (about 10–15 minutes for 3 stores).

Set `DEMO_MODE = True` to test the pipeline on synthetic data without Kaggle.

## Repo structure
```
notebooks/m5_demand_forecasting.ipynb   main notebook (Colab-ready)
notebooks/m5_demand_forecasting.py      same notebook as a plain script (easier to review diffs)
results/                                metrics CSVs, charts and the 28-day forecast
requirements.txt
```

## Next steps
- Item-level forecasts with hierarchical reconciliation (store → state → total)
- Prediction intervals with LightGBM quantile loss, for safety-stock planning
- Scheduled weekly refresh (GitHub Actions) writing forecasts to a dashboard

## Data
[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy), Walmart via Kaggle.
Data is not included in this repo.
