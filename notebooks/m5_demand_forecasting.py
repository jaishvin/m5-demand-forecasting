# %% [markdown]
# # Retail Demand Forecasting: SARIMA vs Prophet vs LightGBM
# **Data:** Walmart M5 (Kaggle) · **Stack:** PySpark, statsmodels, Prophet, LightGBM, pandas, matplotlib
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaishvin/m5-demand-forecasting/blob/main/notebooks/m5_demand_forecasting.ipynb)
#
# **Business question:** how many units will each store × department sell over the next 28 days, and which
# forecasting approach should a demand planning team use?
#
# **Pipeline**
# 1. **Ingest + feature engineering in PySpark:** reshape 30k SKU × 1,941-day wide sales into long format, join the
#    calendar (events, SNAP days) and weekly prices, build discount features with window functions, aggregate to
#    store × department, then create lag and rolling features.
# 2. **Backtest** 4 models with rolling-origin cross-validation (3 folds × 28-day horizon):
#    seasonal naive (baseline), SARIMA, Prophet (events + price regressors), LightGBM (global model, Tweedie loss).
# 3. **Evaluate** with WMAPE, forecast accuracy (1 − WMAPE) and bias, overall, by department and on event days.
# 4. **Forecast** the next 28 days with the best model and save it to CSV.

# %% [markdown]
# ## 1. Setup
# Installs the libraries (Colab) and makes sure Java 17+ is available for Spark.

# %%
import os, re, sys, time, glob, zipfile, logging, warnings, subprocess

IN_COLAB = "google.colab" in sys.modules

def java_major_version():
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
        m = re.search(r'version "(\d+)', out)
        return int(m.group(1)) if m else 0
    except FileNotFoundError:
        return 0

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pyspark>=4.0", "prophet", "lightgbm", "statsmodels", "kaggle"], check=True)
    if java_major_version() < 17:
        print("Installing Java 17 for Spark (about 1 minute)...")
        subprocess.run("apt-get -qq update && apt-get -qq install -y openjdk-17-jdk-headless > /dev/null",
                       shell=True, check=True)
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
        os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

print("Java version:", java_major_version())

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pyspark.sql import SparkSession, functions as F, Window

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True      # silence Prophet's per-fit logs
logging.getLogger("prophet").setLevel(logging.ERROR)
plt.rcParams.update({"figure.figsize": (12, 4), "axes.grid": True, "grid.alpha": 0.3})

# %% [markdown]
# ## 2. Configuration
# `DEMO_MODE = True` generates a small synthetic dataset in the M5 format, so you can check the whole pipeline
# runs in a few minutes before downloading the real data.

# %%
DEMO_MODE = False                    # True = synthetic data (quick test); False = real M5 data from Kaggle
DATA_DIR = "data"
RESULTS_DIR = "results"
STORES = ["CA_1", "TX_1", "WI_1"]    # one store per state; set to None for all 10 stores (slower)
HORIZON = 28                         # forecast 28 days ahead (the M5 competition horizon)
N_FOLDS = 3                          # rolling-origin backtest folds
SARIMA_TRAIN_DAYS = 730              # SARIMA trains on the last 2 years (faster, recent patterns)
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)

# %% [markdown]
# ## 3. Get the data
# **Real data (one-time setup):**
# 1. Sign in at kaggle.com, open the [M5 Forecasting – Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules) and click **I Understand and Accept** on the Rules tab.
# 2. Kaggle → Settings → API → **Create New Token**. It downloads `kaggle.json` with your username and key.
# 3. In Colab, click the key icon (Secrets) on the left and add `KAGGLE_USERNAME` and `KAGGLE_KEY`, with notebook access on.
#
# Alternative: download `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` yourself and upload them to the `data/` folder.

# %%
NEEDED = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]

def download_m5(data_dir):
    if all(os.path.exists(os.path.join(data_dir, f)) for f in NEEDED):
        print("M5 files already present.")
        return
    if IN_COLAB:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    subprocess.run(["kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy", "-p", data_dir],
                   check=True)
    with zipfile.ZipFile(os.path.join(data_dir, "m5-forecasting-accuracy.zip")) as z:
        for f in NEEDED:
            z.extract(f, data_dir)
    print("Downloaded:", NEEDED)


def make_synthetic_m5(data_dir, stores, items_per_dept=5, n_days=1941, n_cal_days=1969, seed=SEED):
    """Writes small CSVs with the same columns as the real M5 files (for testing the pipeline)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-29", periods=n_cal_days, freq="D")
    cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "d": [f"d_{i}" for i in range(1, n_cal_days + 1)]})
    cal["wm_yr_wk"] = 11101 + np.arange(n_cal_days) // 7
    cal["weekday"] = dates.day_name()
    cal["wday"] = (dates.dayofweek + 2) % 7 + 1          # M5 convention: 1 = Saturday
    cal["month"], cal["year"] = dates.month, dates.year
    ev_name, ev_type = [None] * n_cal_days, [None] * n_cal_days
    for i, d in enumerate(dates):
        if d.month == 12 and d.day == 25: ev_name[i], ev_type[i] = "Christmas", "National"
        elif d.month == 7 and d.day == 4: ev_name[i], ev_type[i] = "IndependenceDay", "National"
        elif d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28: ev_name[i], ev_type[i] = "Thanksgiving", "National"
        elif d.month == 2 and d.weekday() == 6 and d.day <= 7: ev_name[i], ev_type[i] = "SuperBowl", "Sporting"
        elif d.month == 10 and d.day == 31: ev_name[i], ev_type[i] = "Halloween", "Cultural"
    cal["event_name_1"], cal["event_type_1"] = ev_name, ev_type
    cal["event_name_2"], cal["event_type_2"] = None, None
    cal["snap_CA"] = (dates.day <= 10).astype(int)
    cal["snap_TX"] = dates.day.isin([1, 3, 5, 6, 7, 9, 11, 12, 15]).astype(int)
    cal["snap_WI"] = dates.day.isin([2, 3, 5, 6, 8, 9, 11, 12, 14, 15]).astype(int)
    cal.to_csv(os.path.join(data_dir, "calendar.csv"), index=False)

    depts = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
    t = np.arange(n_days)
    weekly = np.array([1.25, 1.2, 0.95, 0.85, 0.85, 0.9, 1.0])[np.asarray(dates[:n_days].dayofweek)]
    yearly = 1 + 0.15 * np.sin(2 * np.pi * t / 365.25)
    is_xmas = np.asarray((dates[:n_days].month == 12) & (dates[:n_days].day == 25))
    sales_rows, price_rows = [], []
    weeks = cal["wm_yr_wk"].unique()
    for store in stores:
        state = store.split("_")[0]
        snap = cal[f"snap_{state}"].values[:n_days]
        for dept in depts:
            for k in range(1, items_per_dept + 1):
                item = f"{dept}_{k:03d}"
                base_price = rng.uniform(1, 20)
                promo_weeks = rng.random(len(weeks)) < 0.12
                wk_price = np.where(promo_weeks, base_price * 0.8, base_price).round(2)
                price_rows += [(store, item, int(w), float(p)) for w, p in zip(weeks, wk_price)]
                day_price = wk_price[(cal["wm_yr_wk"].values[:n_days] - 11101)]
                base = rng.uniform(1, 6) * (3 if dept.startswith("FOODS") else 1)
                lam = (base * weekly * yearly * (1 + 0.0002 * t) * (1 + 0.25 * snap * dept.startswith("FOODS"))
                       * (day_price / base_price) ** -1.5)
                units = rng.poisson(lam)
                units[is_xmas] = 0
                sales_rows.append([f"{item}_{store}_evaluation", item, dept, dept.split("_")[0], store, state]
                                  + units.tolist())
    sales = pd.DataFrame(sales_rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
                         + [f"d_{i}" for i in range(1, n_days + 1)])
    sales.to_csv(os.path.join(data_dir, "sales_train_evaluation.csv"), index=False)
    pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        os.path.join(data_dir, "sell_prices.csv"), index=False)
    print(f"Synthetic M5 data written: {len(sales)} items, {n_days} days.")


if DEMO_MODE:
    DATA_DIR = os.path.join(DATA_DIR, "synthetic")
    os.makedirs(DATA_DIR, exist_ok=True)
    make_synthetic_m5(DATA_DIR, STORES or ["CA_1", "TX_1", "WI_1"])
else:
    download_m5(DATA_DIR)

# %% [markdown]
# ## 4. Feature engineering in PySpark
# The raw sales file is **wide** (one column per day). Spark reshapes it to **long** format, joins calendar and
# price data, and builds features with window functions. Working at store × department level gives stable
# series that all four models can be compared on fairly.

# %%
spark = (SparkSession.builder.appName("m5-demand-forecasting").master("local[*]")
         .config("spark.driver.memory", "8g")
         .config("spark.sql.shuffle.partitions", "32")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
t0 = time.time()

sales_wide = spark.read.csv(f"{DATA_DIR}/sales_train_evaluation.csv", header=True)
calendar = spark.read.csv(f"{DATA_DIR}/calendar.csv", header=True, inferSchema=True)
prices = spark.read.csv(f"{DATA_DIR}/sell_prices.csv", header=True, inferSchema=True)

if STORES:
    sales_wide = sales_wide.filter(F.col("store_id").isin(STORES))
    prices = prices.filter(F.col("store_id").isin(STORES))

# 4a. Wide -> long: one row per item per day
id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
sales_long = (sales_wide.unpivot(id_cols, day_cols, "d", "units")
              .withColumn("units", F.col("units").cast("int")))

# 4b. Aggregate items -> store x department x day
dept_daily = sales_long.groupBy("store_id", "state_id", "dept_id", "d").agg(F.sum("units").alias("units"))

# 4c. Price features at item level, then aggregated to department x week
#     discount_depth = price vs the item's highest price in the previous 8 weeks (a promotion proxy)
item_dept = sales_wide.select("item_id", "dept_id").distinct()
w_item = Window.partitionBy("store_id", "item_id").orderBy("wm_yr_wk")
prices = (prices.join(item_dept, "item_id")
          .withColumn("ref_price", F.max("sell_price").over(w_item.rowsBetween(-8, -1)))
          .withColumn("discount_depth",
                      F.least(F.lit(0.0), F.coalesce(F.col("sell_price") / F.col("ref_price") - 1, F.lit(0.0))))
          .withColumn("is_discounted", (F.col("discount_depth") < -0.05).cast("int")))
dept_price = prices.groupBy("store_id", "dept_id", "wm_yr_wk").agg(
    F.avg("sell_price").alias("avg_price"),
    F.avg("discount_depth").alias("avg_discount_depth"),
    F.avg("is_discounted").alias("pct_items_discounted"),
    F.countDistinct("item_id").alias("n_items_listed"))

# 4d. Full grid of series x calendar days (includes the 28 future days, where units are null)
calendar = calendar.select("d", F.to_date("date").alias("date"), "wm_yr_wk", "wday", "month", "year",
                           "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI")
series_keys = sales_wide.select("store_id", "state_id", "dept_id").distinct()
grid = (series_keys.crossJoin(calendar)
        .join(dept_daily, ["store_id", "state_id", "dept_id", "d"], "left")
        .join(dept_price, ["store_id", "dept_id", "wm_yr_wk"], "left")
        .withColumn("series_id", F.concat_ws("_", "store_id", "dept_id"))
        .withColumn("snap", F.when(F.col("state_id") == "CA", F.col("snap_CA"))
                             .when(F.col("state_id") == "TX", F.col("snap_TX"))
                             .otherwise(F.col("snap_WI")))
        .withColumn("is_event", F.col("event_name_1").isNotNull().cast("int"))
        .withColumn("event_type", F.coalesce(F.col("event_type_1"), F.lit("None")))
        .withColumn("day_of_month", F.dayofmonth("date"))
        .withColumn("week_of_year", F.weekofyear("date")))

# 4e. Lag and rolling features. Every lag is >= 28 days, so one model can forecast all 28 days directly
#     (no recursive forecasting, no leakage from the forecast window).
w = Window.partitionBy("series_id").orderBy("date")
grid = grid.withColumn("price_index", F.col("avg_price") / F.avg("avg_price").over(w.rowsBetween(-364, 0)))
for lag in [28, 35, 42, 49, 364]:
    grid = grid.withColumn(f"lag_{lag}", F.lag("units", lag).over(w))
grid = (grid
        .withColumn("roll_mean_7_lag28", F.avg("units").over(w.rowsBetween(-34, -28)))
        .withColumn("roll_mean_28_lag28", F.avg("units").over(w.rowsBetween(-55, -28)))
        .withColumn("roll_std_28_lag28", F.stddev("units").over(w.rowsBetween(-55, -28))))

grid.write.mode("overwrite").parquet(f"{DATA_DIR}/features.parquet")
df = spark.read.parquet(f"{DATA_DIR}/features.parquet").orderBy("series_id", "date").toPandas()
df["date"] = pd.to_datetime(df["date"])
price_cols = ["avg_price", "price_index", "avg_discount_depth", "pct_items_discounted", "n_items_listed"]
df[price_cols] = df.groupby("series_id")[price_cols].transform(lambda s: s.ffill().bfill())
print(f"Spark feature pipeline: {time.time() - t0:.0f}s | {df.series_id.nunique()} series | {len(df):,} rows")
df.head()

# %% [markdown]
# ## 5. Quick EDA

# %%
hist = df[df["units"].notna()]
total = hist.groupby("date")["units"].sum()
fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [3, 1]})
total.plot(ax=ax[0], lw=0.6, title="Total daily units (selected stores)")
ev = hist[hist["is_event"] == 1].groupby("date")["units"].sum()
ax[0].scatter(ev.index, ev.values, s=8, c="tab:red", label="event day", zorder=3)
ax[0].legend()
dow = hist.assign(dow=hist["date"].dt.day_name().str[:3]).groupby("dow")["units"].mean()
dow.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).plot.bar(ax=ax[1], title="Avg units by weekday")
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/eda.png", dpi=120); plt.show()

snap_lift = hist.groupby(["snap", hist["dept_id"].str.split("_").str[0]])["units"].mean().unstack(0)
print("Average units on SNAP vs non-SNAP days by category:\n", (snap_lift[1] / snap_lift[0] - 1).round(3))

# %% [markdown]
# ## 6. Models
# All four models get the same training data and forecast the same 28 days, so the comparison is fair.

# %%
def wmape(y, yhat):
    return np.abs(y - yhat).sum() / np.abs(y).sum()

def bias(y, yhat):
    return (yhat - y).sum() / y.sum()


def seasonal_naive(train, test):
    """Baseline: repeat the last observed week."""
    out = []
    for sid, g in test.groupby("series_id"):
        last_week = train[train.series_id == sid].sort_values("date")["units"].values[-7:]
        out.append(g[["series_id", "date"]].assign(yhat=np.resize(last_week, len(g))))
    return pd.concat(out)


def sarima(train, test):
    """SARIMA(1,0,1)(1,1,1,7) per series on log1p(units)."""
    out = []
    for sid, g in test.groupby("series_id"):
        y = train[train.series_id == sid].sort_values("date")["units"].values[-SARIMA_TRAIN_DAYS:]
        try:
            fit = SARIMAX(np.log1p(y), order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False, maxiter=50)
            yhat = np.expm1(fit.forecast(len(g)))
        except Exception as e:                       # fall back to the baseline if a fit fails
            print(f"SARIMA failed for {sid}: {e}")
            yhat = np.resize(y[-7:], len(g))
        out.append(g[["series_id", "date"]].assign(yhat=np.clip(yhat, 0, None)))
    return pd.concat(out)


PROPHET_REGRESSORS = ["snap", "price_index", "avg_discount_depth"]

def prophet_holidays(frame):
    ev = frame.loc[frame["event_name_1"].notna(), ["date", "event_name_1"]].drop_duplicates()
    return ev.rename(columns={"date": "ds", "event_name_1": "holiday"}).assign(lower_window=-1, upper_window=1)

def prophet_model(train, test):
    """Prophet per series with event holidays and SNAP/price regressors."""
    holidays = prophet_holidays(pd.concat([train, test]))
    out = []
    for sid, g in test.groupby("series_id"):
        tr = train[train.series_id == sid].rename(columns={"date": "ds", "units": "y"})
        te = g.rename(columns={"date": "ds"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="multiplicative", holidays=holidays)
        for r in PROPHET_REGRESSORS:
            m.add_regressor(r)
        m.fit(tr[["ds", "y"] + PROPHET_REGRESSORS])
        yhat = m.predict(te[["ds"] + PROPHET_REGRESSORS])["yhat"].values
        out.append(g[["series_id", "date"]].assign(yhat=np.clip(yhat, 0, None)))
    return pd.concat(out)


LGB_FEATURES = ["store_id", "dept_id", "wday", "month", "day_of_month", "week_of_year", "snap", "is_event",
                "event_type", "avg_price", "price_index", "avg_discount_depth", "pct_items_discounted",
                "n_items_listed", "lag_28", "lag_35", "lag_42", "lag_49", "lag_364",
                "roll_mean_7_lag28", "roll_mean_28_lag28", "roll_std_28_lag28"]
CAT_FEATURES = ["store_id", "dept_id", "event_type"]
LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.03, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  seed=SEED, verbose=-1)
LGB_ROUNDS = 800

def to_lgb(frame):
    X = frame[LGB_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=sorted(df[c].unique()))
    return X

def lightgbm_fit(train):
    tr = train[train["lag_28"].notna()]
    return lgb.train(LGB_PARAMS, lgb.Dataset(to_lgb(tr), tr["units"], categorical_feature=CAT_FEATURES),
                     num_boost_round=LGB_ROUNDS)

def lightgbm_model(train, test, return_model=False):
    """One global LightGBM model across all series (Tweedie loss suits count data)."""
    booster = lightgbm_fit(train)
    pred = test[["series_id", "date"]].assign(yhat=np.clip(booster.predict(to_lgb(test)), 0, None))
    return (pred, booster) if return_model else pred


MODELS = {"Seasonal naive": seasonal_naive, "SARIMA": sarima, "Prophet": prophet_model, "LightGBM": lightgbm_model}

# %% [markdown]
# ## 7. Backtest: rolling-origin cross-validation
# Each fold trains on everything up to a cutoff date and forecasts the next 28 days. Three folds give a fairer
# estimate than a single train/test split, because one month can be unusual.

# %%
history = df[df["units"].notna()].copy()
last_date = history["date"].max()
cutoffs = [last_date - pd.Timedelta(days=HORIZON * k) for k in range(N_FOLDS, 0, -1)]

preds, timings = [], {}
for fold, cutoff in enumerate(cutoffs, 1):
    train = history[history["date"] <= cutoff]
    test = history[(history["date"] > cutoff) & (history["date"] <= cutoff + pd.Timedelta(days=HORIZON))]
    for name, fn in MODELS.items():
        t0 = time.time()
        p = fn(train, test).merge(test[["series_id", "date", "units", "dept_id", "is_event"]],
                                  on=["series_id", "date"])
        preds.append(p.assign(model=name, fold=fold, cutoff=cutoff.date()))
        timings[name] = timings.get(name, 0) + time.time() - t0
    print(f"Fold {fold}: train to {cutoff.date()}, test {test.date.min().date()} to {test.date.max().date()}")

preds = pd.concat(preds, ignore_index=True)
print("Total fit + predict time (s):", {k: round(v) for k, v in timings.items()})

# %% [markdown]
# ## 8. Results

# %%
def score(g):
    return pd.Series({"WMAPE": wmape(g.units, g.yhat), "Bias": bias(g.units, g.yhat)})

by_fold = preds.groupby(["model", "fold"]).apply(score).reset_index()
summary = (by_fold.groupby("model")[["WMAPE", "Bias"]].mean()
           .assign(**{"Forecast accuracy (1-WMAPE)": lambda x: 1 - x["WMAPE"]})
           .sort_values("WMAPE"))
baseline = summary.loc["Seasonal naive", "WMAPE"]
summary["Improvement vs baseline"] = 1 - summary["WMAPE"] / baseline
summary["Fit time (s)"] = pd.Series(timings).round(0)
summary.to_csv(f"{RESULTS_DIR}/backtest_summary.csv")
by_fold.to_csv(f"{RESULTS_DIR}/backtest_by_fold.csv", index=False)
BEST = summary.index[0]
print(f"Best model: {BEST}")
summary.style.format({"WMAPE": "{:.1%}", "Bias": "{:+.1%}", "Forecast accuracy (1-WMAPE)": "{:.1%}",
                      "Improvement vs baseline": "{:.1%}"})

# %%
order = summary.index.tolist()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(order[::-1], summary["WMAPE"][::-1], color=["#9aa5b1"] * (len(order) - 1) + ["#2a6fdb"])
for model, g in by_fold.groupby("model"):
    ax.scatter(g["WMAPE"], [order[::-1].index(model)] * len(g), color="black", s=12, zorder=3)
fold_max = by_fold.groupby("model")["WMAPE"].max()
for i, model in enumerate(order[::-1]):
    ax.text(fold_max[model], i, f"   {summary.loc[model, 'WMAPE']:.1%}", va="center", fontweight="bold")
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlim(0, fold_max.max() * 1.2)
ax.set_xlabel("WMAPE (lower is better) · bars = mean, dots = folds")
ax.set_title("28-day forecast error by model")
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=120); plt.show()

# %%
dept_scores = (preds.groupby(["model", "dept_id"]).apply(lambda g: wmape(g.units, g.yhat))
               .unstack(0)[order])
event_scores = (preds.groupby(["model", "is_event"]).apply(lambda g: wmape(g.units, g.yhat))
                .unstack(1).rename(columns={0: "Normal days", 1: "Event days"}).loc[order])
dept_scores.to_csv(f"{RESULTS_DIR}/wmape_by_department.csv")
print("WMAPE by department:"); display(dept_scores.style.format("{:.1%}").highlight_min(axis=1))
print("WMAPE on event vs normal days:"); display(event_scores.style.format("{:.1%}"))

# %%
last_fold = preds[preds.fold == N_FOLDS]
top_series = history.groupby("series_id")["units"].sum().idxmax()
s = last_fold[last_fold.series_id == top_series]
ctx = history[(history.series_id == top_series) & (history.date > cutoffs[-1] - pd.Timedelta(days=56))
              & (history.date <= cutoffs[-1])]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(ctx.date, ctx.units, color="black", lw=1, label="history")
ax.plot(s[s.model == BEST].date, s[s.model == BEST].units, color="black", lw=2, label="actual")
for name in order:
    sm = s[s.model == name]
    ax.plot(sm.date, sm.yhat, lw=2 if name == BEST else 1, ls="-" if name == BEST else "--", label=name)
ax.axvline(cutoffs[-1], color="grey", ls=":")
ax.set_title(f"{top_series}: last backtest fold"); ax.legend(ncol=3)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/forecast_example.png", dpi=120); plt.show()

# %% [markdown]
# ## 9. What drives the LightGBM forecast?

# %%
_, booster = lightgbm_model(history, history.tail(1), return_model=True)
imp = (pd.Series(booster.feature_importance("gain"), index=LGB_FEATURES)
       .sort_values().tail(15) / booster.feature_importance("gain").sum())
imp.plot.barh(figsize=(8, 5), title="LightGBM feature importance (share of total gain)", color="#2a6fdb")
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=120); plt.show()

# %% [markdown]
# ## 10. Forecast the next 28 days
# Retrains the best model on all history and forecasts the 28 days after the last actual date.

# %%
future = df[df["date"] > last_date].copy()
forecast = MODELS[BEST](history, future)
forecast = forecast.merge(future[["series_id", "date", "store_id", "dept_id"]], on=["series_id", "date"])
forecast = forecast.rename(columns={"yhat": "forecast_units"}).assign(model=BEST)
forecast["forecast_units"] = forecast["forecast_units"].round(1)
forecast.to_csv(f"{RESULTS_DIR}/forecast_next_28_days.csv", index=False)

recent = history[history.date > last_date - pd.Timedelta(days=90)].groupby("date")["units"].sum()
fut = forecast.groupby("date")["forecast_units"].sum()
ax = recent.plot(color="black", lw=1, label="actual")
fut.plot(ax=ax, color="#2a6fdb", lw=2, label=f"forecast ({BEST})")
ax.set_title("Total units: last 90 days and next 28-day forecast"); ax.legend()
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/next_28_days.png", dpi=120); plt.show()
print(forecast.groupby("dept_id")["forecast_units"].sum().round(0))

# %% [markdown]
# ## 11. Summary
# Fill this in from your own run (the numbers come from the tables above):
#
# - **Best model:** ___ with WMAPE ___% (forecast accuracy ___%), ___% better than the seasonal-naive baseline.
# - **SARIMA vs ML:** SARIMA scored ___%. LightGBM's lead comes from ___ (see feature importance).
# - **Event days:** error rises from ___% on normal days to ___% on event days. Next step: more event features.
# - **Next steps:** item-level forecasts with hierarchical reconciliation, prediction intervals (quantile loss),
#   and a scheduled weekly refresh.

# %%
spark.stop()
