# Wafer Map Defect Pattern Classification (WM-811K)

Machine-learning pipeline that classifies spatial defect signatures on semiconductor
wafer maps — the patterns (edge rings, scratches, center clusters, …) that point a
defect engineer toward a specific tool or process step during excursion response.

Built on the public **WM-811K / LSWMD** dataset: 811,457 real wafer maps from
46,293 production lots, ~172k of them labeled with one of 9 failure classes
(Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none).

## Why this problem

In a fab, the *spatial distribution* of failing die is a fingerprint of the root
cause: an edge ring implicates edge-exclusion / chamber uniformity, a scratch
implicates wafer handling, a repeating cluster implicates a reticle or chuck.
Classifying these signatures automatically shortens time-to-root-cause during
excursions and removes a manual, subjective review step. This project mirrors
that task end-to-end on public data: heavy class imbalance, mostly-unlabeled
maps, and variable wafer geometries — the same properties real fab data has.

## Approach

1. **EDA** — label distribution, wafer geometry survey, per-class signature gallery
   (`notebooks/01_eda.ipynb`)
2. **Classical baseline** — density / geometry / radon-projection features into a
   gradient-boosted classifier (`notebooks/02_baseline.ipynb`)
3. **CNN** — small convolutional network on normalized 64×64 wafer maps, trained
   with class-balanced sampling (`notebooks/03_cnn.ipynb`)
4. **Evaluation** — per-class precision/recall and confusion analysis, with
   discussion of which confusions matter in a fab context and which don't

## Results

Test-set performance (9 classes, macro-averaged):

| Model | Test mix | Macro F1 | "none" F1 | Notes |
|---|---|---|---|---|
| Engineered features + HistGradientBoosting | "none" capped at 20k | 0.84 | 0.946 | Strong on geometric signatures (Edge-Ring 0.96 F1) |
| Small CNN (~300k params, class-balanced sampling) | Full production mix (85% "none") | 0.83 | 0.987 | Near-eliminates false alarms at realistic class ratios |

Headline finding: at production class ratios the CNN's value is **false-alarm
suppression** (0.989 precision on clean wafers) rather than raw accuracy —
the property that decides whether an auto-classifier saves reviewer time or
buries engineers in noise. Full analysis in `notebooks/03_cnn.ipynb`.

## Repo layout

```
src/            reusable pipeline code (data loading, features, models, training)
notebooks/      narrative analysis — start with 01_eda.ipynb
reports/        exported figures
data/           dataset (not committed — see below)
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Dataset: http://mirlab.org/dataSet/public/MIR-WM811K.zip  (~330 MB)
# Unzip so that data/raw/MIR-WM811K/ exists, then:
python -m src.prepare_data
jupyter lab
```

## Dataset citation

Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). *Wafer Map Failure Pattern
Recognition and Similarity Ranking for Large-Scale Data Sets.* IEEE Transactions
on Semiconductor Manufacturing, 28(1), 1–12.
