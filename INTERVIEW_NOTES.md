# Interview notes: defending every design choice

## 60-second pitch
"At Myntra I forecast demand with SARIMA, and I wanted to know how much ML adds and why. I took Walmart's M5
data, about 30,000 SKUs over 5 years, and built a PySpark pipeline that reshapes the data, joins prices and
events, and creates lag, rolling and discount features. Then I backtested four models over three rolling
28-day windows. [Best model] reached [X]% WMAPE, [Y]% better than the seasonal-naive baseline, with
[top features] driving most of the gain. The biggest remaining error is on event days, which is where I'd
invest next."

## Likely questions

**Why PySpark when the data fits in pandas?**
The raw data is about 59M rows in long format across all stores, and in production it would be much larger.
Spark's window functions (lags, rolling stats per series) and lazy execution scale out without changing the code.
I aggregate down to store × department in Spark, then model in pandas, where the libraries are.

**Why WMAPE and not MAPE?**
MAPE blows up when actuals are near zero, which is common in retail, and it weights all series equally.
WMAPE = Σ|actual − forecast| / Σ actual weights errors by volume, so high-selling departments count more.
That matches business impact. Forecast accuracy = 1 − WMAPE.

**Why rolling-origin CV and not a random train/test split?**
A random split leaks future information into training. Rolling-origin trains on everything up to a cutoff and
tests on the next 28 days, repeated 3 times. That mimics how the model is used, and avoids judging it on one
unusual month.

**Why are all lags ≥ 28 days?**
The horizon is 28 days. A 7-day lag would need actuals we won't have on day 8 onwards of the forecast. With
lags ≥ 28, one model forecasts the whole horizon directly: no recursive error build-up, no leakage.

**Why Tweedie loss in LightGBM?**
Sales are non-negative counts with many low values and a long right tail. Tweedie (power 1.1, close to
Poisson) fits that distribution better than squared error, which can predict negatives and over-weights spikes.

**Why one global LightGBM model and not one per series?**
It learns shared patterns (weekday effect, price response) across all series. It has more data per pattern
and handles new or short series better. Store and department are categorical features.

**SARIMA order (1,0,1)(1,1,1,7)?**
Weekly seasonality (m = 7) in daily retail data. Seasonal differencing D = 1 removes the weekly pattern.
The AR and MA terms capture short-term autocorrelation. I fit it on log1p(units) to stabilise variance.
The order could be tuned with AIC (auto_arima); I kept it fixed for a fair, fast comparison.

**How does Prophet use events and price?**
Calendar events go in as holidays with a ±1-day window. SNAP, price index and discount depth go in as
extra regressors. Multiplicative seasonality, because the weekly swing grows with the level.

**What's the discount-depth feature?**
For each item and week: price ÷ the item's highest price in the previous 8 weeks, minus 1, capped at 0.
For example, −0.2 means 20% below its recent regular price. It's a promotion proxy, since M5 has no promo
flag. This is like how markdowns show up in EORS/BBD sale periods.

**What would you do with more time?**
Item-level forecasts reconciled up the hierarchy, prediction intervals for safety stock (quantile loss),
hyperparameter tuning with Optuna inside time-series CV, and a scheduled weekly retrain.

**How does this connect to your Myntra work?** *(Keep it to what you've actually done)*
Same problem and metric (forecast accuracy), and the same use: buying and replenishment decisions. In this
project I tested ML and price/event features beyond the SARIMA approach, and that is what I'd bring to a
forecasting role.

## Numbers to memorise after your run
- WMAPE for each model: ___
- % improvement of the best model over the baseline: ___
- Top 3 features: ___
- Event-day vs normal-day WMAPE: ___
- Spark pipeline runtime and row count: ___
