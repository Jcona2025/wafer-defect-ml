# Wafer Map Triage

An ML tool that reads wafer maps, names the failure signature, and points at a
root-cause direction — the call a defect engineer makes by eye, automated on
every wafer.

![Lot triage view](reports/figures/app_lot.png)

## Run it

```bash
docker compose -f deploy/docker-compose.yml up -d   # then open :8501
```

or locally: `pip install -r requirements-app.txt && streamlit run app.py`
(a demo dataset and trained model ship in the repo — no setup needed).

Three views: **Single wafer** (classify + diagnose one map), **Lot triage**
(25 wafers in one pass — auto-clear clean, flag signatures, queue ambiguous ones
for review), **Cluster view** (CNN embedding of wafer maps — similar failures
group together; novel signatures appear as new clusters).

## Data

**WM-811K**: 811,457 wafer maps from 46,293 production lots of a real fab
(Wu, Jang & Chen, IEEE Trans. Semiconductor Manufacturing, 2015 —
[download](http://mirlab.org/dataSet/public/)). Each map is a die grid of
0 = outside wafer / 1 = pass / 2 = fail. 172,950 maps carry an engineer-assigned
label across 9 signature classes; 85% of those are "none" — the same imbalance
production data has.

## ML

Maps are resampled to 64×64 and one-hot encoded (outside/pass/fail). Two models,
compared deliberately:

- **Baseline** — 18 engineered features (radial ring densities, projection
  statistics, region geometry) + gradient boosting. Interpretable in fab language.
- **CNN** — 3 conv blocks, 289k parameters, trained with class-balanced sampling
  so rare signatures aren't drowned out by the 85% clean majority.

## Results

Held-out test set, 34,590 wafers at true production class mix:

| | Baseline | CNN |
|---|---|---|
| Macro F1 (9 classes) | 0.84¹ | **0.83** |
| Clean-wafer precision ("none") | 0.973 | **0.989** |
| Edge-Ring F1 | 0.960 | **0.972** |
| Scratch precision | 0.509 | **0.743** |

¹ baseline tested with the clean class capped; the CNN faces the full 85%-clean mix.

The number that decides usefulness is **clean-wafer precision**: at production
ratios, the CNN false-flags fewer than 1 in 90 clean wafers — the difference
between an alarm engineers trust and one they mute. Weakest classes for both
models are small diffuse clusters (Loc) and faint scratches — genuinely ambiguous
at map resolution; the tool routes low-confidence wafers to human review instead
of guessing.

Full analysis: `notebooks/00_technical_summary.ipynb` (condensed, executed) and
`01`–`03` (EDA → baseline → CNN).
