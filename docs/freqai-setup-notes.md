# FreqAI Setup Notes (GT-Quant)

Hard-won configuration facts from the Day 1–2 bring-up. Check these first when
FreqAI produces zero trades or zero predictions.

## Gotchas hit during bring-up

1. **`include_timeframes` cannot be smaller than the main timeframe.**
   Main TF is 5m, so `1m` is rejected. 1m microstructure comes from the
   Cryptofeed collector (Day 2 live), not from FreqAI.

2. **FreqAI needs the `freqai` Docker image.**
   `freqtradeorg/freqtrade:stable` lacks `datasieve` → ModuleNotFoundError.
   Use `freqtradeorg/freqtrade:stable_freqai`.

3. **`self.freqai.start(dataframe, metadata, self)` is REQUIRED** at the end of
   `populate_indicators`. Without it FreqAI never trains and never predicts —
   silently. No error, just `do_predict` absent and zero trades.

4. **Single-target models reject multiple `&-` labels** with
   "DataFrame for label cannot have multiple columns". Run separate FreqAI
   *identifiers* per prediction horizon (per plan Day 3) or use a
   `*MultiTarget` model class.

5. **`model_training_parameters` must match the model class.**
   XGBoost params (`objective: reg:squarederror`) passed to LightGBMRegressor
   kill training with "Unknown objective type name" — the exception is caught
   per window, training is *skipped*, and predictions are **zero-filled**.
   Symptom: predictions all exactly 0.0, `do_predict` all 0, no model files
   in `sub-train-*` dirs, but no fatal error.

6. **Stale backtest predictions are REUSED.**
   If `models/<identifier>/backtesting_predictions/` exists, FreqAI loads it
   ("Found backtesting prediction file") instead of retraining — even if those
   predictions are zeros from a failed run. Purge the identifier dir after any
   failed experiment:
   `rm -rf user_data/models/<identifier>`.

7. **`--cache=day` (default) reuses analyzed dataframes** from a previous
   backtest of the same strategy+timerange, so `populate_indicators` (and
   therefore `freqai.start`) never re-runs. Use `--cache none` while iterating.

8. **Prediction columns are absent on some `populate_entry/exit_trend` calls.**
   Guard with `if col not in dataframe.columns: return dataframe` before using
   `&-` / `do_predict` columns, otherwise `.get(col, 0)` returns a scalar and
   `.loc` assignment raises "cannot use a single bool to index into setitem".

## Smoke-test recipe (known good)

```bash
cd ft_userdata
docker-compose run --rm --no-deps freqtrade backtesting \
  --strategy GTQuantMultiTF \
  --freqaimodel LightGBMRegressor \
  --timerange 20260601-20260701 \
  --cache none
```

Healthy signs: "Training N timeranges", no "raised exception", model files
under `user_data/models/<identifier>/sub-train-*/`, non-zero prediction
variance in `backtesting_predictions/*.feather`.

## Current state (Day 2 skeleton)

- 969 features generated per pair (5 TFs × periods × corr pairs × shifts).
- First real backtest (June 2026, LightGBMRegressor): 473 trades,
  ~57% win rate, roughly -1% net after fees — expected pre-tuning.
- Tuning (thresholds, hyperopt, fee erosion) is Day 3–4 work.
