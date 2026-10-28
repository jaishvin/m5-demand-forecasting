# Build guide: from zero to a live GitHub project

Plan for about 7–10 days at 1–2 hours a day. Don't skip the learning steps (days 3–6). The aim is that you can
explain every line in an interview.

## Day 1: Put the repo on GitHub (about 20 min, no coding)
1. Unzip `m5-demand-forecasting.zip` on your Mac.
2. On github.com (signed in as `jaishvin`): **+ → New repository** → name it `m5-demand-forecasting` →
   Public → leave "Add a README" unticked → **Create repository**.
3. On the empty repo page, click **uploading an existing file**. Drag in everything inside the unzipped folder
   (README.md, requirements.txt, .gitignore, `notebooks/`, `docs/`, `results/`) → **Commit changes**.
   On a Mac, press Cmd+Shift+. in Finder to see the hidden `.gitignore` file.
4. Check that the **Open in Colab** badge in the README opens the notebook. It only works if the repo is
   named exactly `m5-demand-forecasting`.

## Day 2: Run it in Colab
1. Click the Colab badge, then **File → Save a copy in Drive** (so your edits are saved).
2. **Quick test:** set `DEMO_MODE = True`, then **Runtime → Run all**. It takes about 3 minutes, and every cell
   should finish without errors.
3. **Real data:**
   - kaggle.com → sign in → open the [M5 rules page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)
     → **I Understand and Accept**.
   - Kaggle → your profile → **Settings → API → Create New Token**. It downloads `kaggle.json`; open it to see your
     username and key.
   - In Colab, click the key icon on the left (Secrets) → add `KAGGLE_USERNAME` and `KAGGLE_KEY` → switch on
     "Notebook access".
   - Set `DEMO_MODE = False` → **Runtime → Run all** (about 10–15 minutes).
4. Download the outputs: in the Colab file panel, right-click `results/` → download each PNG and CSV.

## Days 3–6: Understand it (the part that matters for interviews)
Work through one section a day. Change something, rerun it, and write down what happened.

| Day | Section | Experiment to try |
|---|---|---|
| 3 | PySpark pipeline (section 4) | Print `sales_long.count()` and `grid.printSchema()`. Swap `unpivot` for the older `stack()` expression. Add a `lag_7` feature and explain why it would leak. |
| 4 | SARIMA + Prophet | Try SARIMA order (2,0,1). Remove Prophet's holidays and see how WMAPE changes on event days. |
| 5 | LightGBM | Switch `objective` to `"regression"` and compare with Tweedie. Remove the price features and check the change in WMAPE. |
| 6 | Backtest + metrics | Compute WMAPE by hand for one series in Excel. Change `N_FOLDS` to 5. |

Note your findings; they become README content and interview stories.

## Day 7: Polish and publish
1. In GitHub, open the `results/` folder → **Add file → Upload files** → upload the PNGs and CSVs from your real run.
2. Upload your Colab notebook, with outputs, over `notebooks/m5_demand_forecasting.ipynb`
   (Colab: **File → Download → .ipynb**). Recruiters can then see the charts without running anything.
3. Fill in the **Results** table and **Key findings** in README.md using your real numbers. Edit it on GitHub with
   the pencil icon.
4. Pin the repo on your GitHub profile (Profile → Customize your pins).
5. Update the resume bullet with your real WMAPE numbers.

## Optional stretch (days 8–10)
- Run on all 10 stores (`STORES = None`). Spark handles about 59M rows; this is a good PySpark talking point.
- Add LightGBM quantile models (alpha 0.1 and 0.9) for prediction intervals.
- Add a GitHub Actions workflow that reruns the forecast weekly (this links to project 2, API automation).

## If something breaks
| Problem | Fix |
|---|---|
| `403 Forbidden` from Kaggle | You haven't accepted the competition rules (Day 2, step 3). |
| `KAGGLE_USERNAME` is None | The secret name is misspelled, or "Notebook access" is off. |
| Java / `JAVA_GATEWAY_EXITED` error | Rerun the setup cell; it installs Java 17. Then Runtime → Restart session and Run all. |
| Colab runs out of RAM | Keep `STORES` to 3 stores, or Runtime → Change runtime type → High-RAM. |
| Colab disconnects | Keep the tab open. The free tier disconnects after about 90 minutes idle. |
