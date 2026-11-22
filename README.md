# EXO110R

A small, beginner-friendly, Keras-based exoplanet **candidate ranker**, inspired by Google's AstroNet
([Shallue & Vanderburg, arXiv:1712.05044](https://arxiv.org/abs/1712.05044)).

When a planet passes in front of its star, the star dims by a tiny fraction. EXO110R trains a small 1D CNN on
Kepler brightness records (the Kaggle *Kepler Labelled Time Series* data) to score how planet-like each star's record
looks, and then ranks the stars by that score. AstroNet's authors used a deep-learning model of this kind to help find
Kepler-90i, the eighth planet of its system. **This project does not find new planets**, and it is a much simpler model
than AstroNet.

> **Important:** this is a learning and reproduction project. It does **not** detect new planets and does **not**
> confirm anything. Its output is a **ranked list of candidates**: stars whose brightness record *looks* planet-like
> and might deserve a closer look. Confirmation needs extra observations and expert review.

---

## Run summary

<!-- Paste the output of `python -m exoplanet.summary` (also saved in results/final_summary.txt) inside this block. -->

```text
Run: seed=42, preprocess_v1, model=cnn

Split (by star):        train    val   test
  positives (planets)         30      7      5
  negatives (no planet)     4039   1011    565
  (the final model was trained on train + val together)

Threshold (chosen by cross-validation, never on test data): 0.698

Test results (locked test file, used for the final exam):
  Precision 0.667   Recall 0.400   PR-AUC 0.639   ROC-AUC 0.993
  Confusion matrix: TN=564  FP=1  FN=3  TP=2

Top 10 ranked candidates (held-out test stars):
   1. exoTest.csv row 4    score 0.868   dataset label: planet     (flagged)
   2. exoTest.csv row 241  score 0.863   dataset label: non-planet (flagged)
   3. exoTest.csv row 3    score 0.840   dataset label: planet     (flagged)
   4. exoTest.csv row 2    score 0.557   dataset label: planet     (below threshold)
   5. exoTest.csv row 549  score 0.466   dataset label: non-planet (below threshold)
   6. exoTest.csv row 124  score 0.423   dataset label: non-planet (below threshold)
   7. exoTest.csv row 96   score 0.371   dataset label: non-planet (below threshold)
   8. exoTest.csv row 1    score 0.366   dataset label: planet     (below threshold)
   9. exoTest.csv row 136  score 0.360   dataset label: non-planet (below threshold)
  10. exoTest.csv row 185  score 0.356   dataset label: non-planet (below threshold)

Notes:
  - The Kaggle data has no star names (such as KIC numbers), so stars are identified by row number.
  - Only 5 planets are in the test set, so treat every number as rough.
  - These are ranked candidates, not discoveries or confirmed planets.

```

---

## 1. What the project does (and does not do)

Given a star's brightness record (one row of 3,197 measurements), a small 1D CNN outputs a **score between 0 and 1**
for how much the record resembles the labeled planet stars. Stars are then ranked by score.

**One-line summary:** raw brightness record, then detrend and normalize, then small 1D CNN, then a score, then a ranking.

It does **not**:

- watch stars live or search for dips itself;
- use a period or epoch, and it does **not** fold the light curve (the dataset does not provide them);
- build AstroNet's separate global and local views, or use two network branches;
- download anything from NASA (see section 9 on Lightkurve);
- tell a confirmed planet from a candidate. The labels are the dataset's labels, and every output is a candidate.

---

## 2. Input

The data is the Kaggle dataset
[Kepler Labelled Time Series Data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data).
It has two files, `exoTrain.csv` and `exoTest.csv`. One row is one star.

| Item | What the files contain |
|---|---|
| Label | 1 = no planet, 2 = planet (converted to 0 and 1 in the code) |
| Flux columns | 3,197 brightness measurements per star, already prepared by the dataset's author |
| Training file | 5,087 stars, 37 planets, 5,050 non-planets |
| Test file | 570 stars, 5 planets, 565 non-planets |
| Star ID | **none.** Stars are identified only by their row number in the CSV |
| Period, epoch | **not provided** |
| Mission and cadence | not stored per row. The dataset is described as derived mainly from Campaign 3 observations by NASA's Kepler space telescope (campaigns are K2-mission terminology). Check the Kaggle data card for the exact wording and cadence. |
| Flux product (SAP / PDCSAP) | not stated in the files |

Missing values are never filled in silently. The code stops with an error if it finds one (there are none).

---

## 3. How it works (what was actually built)

```text
raw flux CSV -> detrend + normalize each star -> stratified split by star -> baseline + CNN
             -> cross-validation -> error analysis -> one final test -> ranked lists
```

1. **Load** (`data.py`): read the CSV, turn labels into 0 and 1.
2. **Detrend:** a median filter (window 101 steps) estimates each star's slow drift, which is subtracted. Short dips survive.
3. **Normalize:** subtract each star's median, divide by a robust noise estimate (MAD). Each star is processed on its own.
4. **Clip:** cut only large upward spikes (above 4 noise units). Dips go downward, so they are untouched.
5. **Split by star** (`preprocess.py`): the 5,087 labeled stars are split 80/20 into train and validation, keeping the planet
   share equal. The test file stays locked. There is one light curve per star, so no star can appear in two splits.
6. **Baseline** (`baseline.py`): logistic regression on summary statistics and low-frequency FFT features.
7. **CNN** (`models.py`, `cnn.py`): three `Conv1D` layers with max pooling, global max pooling, dropout, a small dense
   layer and a sigmoid output. Class weights handle the imbalance. No augmentation in the final run.
8. **Cross-validation** (`cv.py`): 5 folds over the 5,087 labeled stars compare the baseline and the CNN, and choose the
   threshold from out-of-fold predictions (never from test data).
9. **Error analysis** (`errors.py`): plots of the most confident false alarms and misses.
10. **Final test** (`final_test.py`): train on all labeled stars, evaluate once on the test file, write the ranked lists.
    The script refuses to run again if `results/final_test.json` exists.

---

## 4. How to run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Download the two CSV files from Kaggle and put them in `data/raw/` (they are not part of this repository).

```bash
python -m exoplanet.preprocess      # clean, split -> data/processed/dataset_v1.npz
python -m exoplanet.baseline        # logistic-regression baseline (single split)
python -m exoplanet.cnn             # CNN (single split)
python -m exoplanet.cv              # 5-fold cross-validation -> results/cv_results.json, cv_oof.npz
python -m exoplanet.errors          # plots of mistakes
python -m exoplanet.final_test      # final exam + ranked lists: run ONCE
python -m exoplanet.summary         # one-page summary -> results/final_summary.txt
python -m pytest -q                 # automatic checks
```

Run `cv` before `errors` and `final_test`, because both read its results.

---

## 5. Output

### In the terminal

`python -m exoplanet.summary` prints the run settings, the data split, the threshold, the test results, the confusion
matrix and the top-ranked stars (see **Run summary** above).

### Is each star processed once, or can it be reused?

- **Every star is processed independently**, but in one batch. A run covers all stars, one row each.
- **There is no per-star cache.** `preprocess` writes one file, `data/processed/dataset_v1.npz`, with all stars. Running it
  again overwrites that file. Settings are stored in `configs/preprocess_v1.json`, and a new version of the settings
  should get a new name (`v2`).
- **Runs are not saved in separate folders.** Results go to `results/`, and the one-time final test is protected by
  `results/final_test.json`.

---

## 6. How data is stored

```text
EXO110R/
  data/                         # gitignored: never commit raw or large data
    raw/        exoTrain.csv, exoTest.csv
    processed/  dataset_v1.npz   # all arrays, labels and the split indices
  configs/preprocess_v1.json
  src/exoplanet/                 # data, preprocess, features, models, baseline, cnn, cv, errors,
                                 # final_test, evaluate, summary
  tests/test_data.py
  results/                       # metrics, plots and ranked lists
  docs/                          # guides (see Documentation below)
```

---

## 7. How ranked results are stored

| File | Contents |
|---|---|
| `results/ranked_candidates_test.csv` | the 570 held-out test stars, most to least planet-like |
| `results/ranked_candidates_labeled_cv.csv` | the 5,087 labeled stars, each scored by a cross-validation model that never trained on it |
| `results/cv_results.json` | cross-validation scores and the chosen threshold |
| `results/final_test.json` | the one-time test results |
| `results/final_summary.txt` | the one-page summary |
| `results/*.png` | example curves, preprocessing check, error gallery |

Columns in the ranked lists: `rank`, `source`, `csv_row` (0-based data row in the original CSV, header not counted),
`planet_score`, `flagged_as_candidate` (score at or above the cross-validated threshold) and `known_label` (the dataset's
label, shown so the ranking can be checked).

A high score means "looks planet-like". It is **not** the probability that the star hosts a confirmed planet. Every star in
this dataset already has a label, so the lists show how high the model puts the known planets. They are not discoveries.

---

## 8. Results and an honest assessment of the model

**5-fold cross-validation** on the 5,087 labeled stars (37 planets). Scores are mean ± spread across folds. Precision and
recall use the threshold that maximizes F1 on the pooled out-of-fold predictions.

| Model | PR-AUC | ROC-AUC | Precision | Recall | Threshold |
|---|---|---|---|---|---|
| Logistic regression baseline | 0.054 ± 0.026 | 0.715 ± 0.074 | 0.114 | 0.135 | 0.998 |
| 1D CNN | 0.193 ± 0.114 | 0.873 ± 0.093 | 0.239 | 0.432 | 0.698 |

A random model scores a PR-AUC of about 0.007 (the planet fraction).

**Final test** (locked test file, 570 stars, 5 planets), CNN trained on all labeled stars, threshold 0.698:

| Metric | Value |
|---|---|
| PR-AUC / ROC-AUC | 0.639 / 0.993 |
| Planets found / missed | 2 / 3 |
| False alarms | 1 |
| Precision / recall | 0.667 / 0.400 |
| Known planets in the top 5 / 10 / 20 of the ranked list | 3 / 4 / 5 |

### What I can and cannot honestly say

- **The CNN is better than the baseline, but it is a modest model.** In cross-validation it found about 43% of the planets
  (16 of 37), and about three out of four stars it flagged were not planets (51 false alarms).
- **The scores are shaky.** The CNN's PR-AUC ranged from 0.04 to 0.37 across folds, and it lost to the baseline in one of
  five folds. It was trained with one seed and 30 fixed epochs, with no tuning.
- **Do not over-read the test result.** With only 5 test planets, one planet changes recall by 20 percentage points. The
  test PR-AUC (0.639) is higher than the cross-validation PR-AUC (0.193), most likely because of the tiny test set, so the
  cross-validation numbers are the more reliable estimate. The final test was run twice with identical settings, the second
  time to add the ranked list.
- **What it gets wrong.** Its most confident false alarms were stars with very deep, repeating dips (some possibly two
  stars eclipsing each other; I could not confirm that). It seems to treat "a big repeating dip" as a planet and cannot
  tell a planet-sized dip from a star-sized one. Most missed planets were extremely noisy curves, and one clean three-dip
  planet still scored low. Some planet and non-planet curves look alike to the eye, which points to hard data and imperfect
  labels, not only a weak model.
- **It has no period, folding or secondary-eclipse checks**, so it cannot do what AstroNet does to separate planets from
  eclipsing binaries.
- **It is not validated on other stars, other missions or new data**, and its scores are rankings, not probabilities.
- **The threshold is a choice.** It maximizes F1. A search that cares more about catching planets, or about avoiding false
  alarms, would pick a different cut-off.

---

## 9. Why Lightkurve is not used

[Lightkurve](https://docs.lightkurve.org/) is a tool for searching NASA's archives, downloading Kepler and TESS light
curves, removing flagged points, detrending and folding them on a period. This project does not use it, for these reasons:

- **The Kaggle files are already a table of brightness numbers** (one row per star, 3,197 columns), so there was nothing
  to download and no archive search to do.
- **The dataset has no period or epoch**, which are needed to fold a light curve. Without them, Lightkurve's folding had
  nothing to work on.
- **A few lines of our own cleaning** (median-filter detrending, robust scaling, upward-only clipping) were enough for this
  data, and are easy to read, test and explain.

This was a scope decision, not a technical limit, and it has a cost: the research guide's first milestone, the Lightkurve
tutorial *Recover a known planet*, was **not completed**, and the project has no folded global and local views.

A future version closer to AstroNet would download light curves with Lightkurve, take periods, epochs and labels from the
NASA Exoplanet Archive, fold each signal, build global and local views, and use a two-branch CNN.

---

## 10. Reproducibility checklist

| Question | Answer |
|---|---|
| Flux product and cadence | as provided in the Kaggle files; not stated in the files, check the data card |
| Quality flags, detrending, clipping | quality flags already applied by the dataset author (assumed); median filter (window 101), upper clip at 4 noise units. See `configs/preprocess_v1.json` |
| How periods and transit centers were obtained | not used |
| Split made by star before creating views | yes (one light curve per star; no views are created) |
| Threshold and the metric used to choose it | 0.698, maximum F1 on pooled out-of-fold cross-validation predictions |
| Final test | CNN, 30 epochs, no augmentation, seed 42; run twice with identical settings |
| Data | Kaggle *Kepler Labelled Time Series Data*, downloaded on `<fill: date>` |
| Python and packages | Python 3.12 (python.org installer); exact package versions in `requirements-lock.txt` |
| Hardware | `<fill: Mac model>`, CPU only |
| Code commit | `<fill: output of git rev-parse --short HEAD>` |

---

## 11. What is used

**Language and libraries**
- Python 3.12
- NumPy, pandas, SciPy: arrays, tables, the median filter and statistics
- scikit-learn: stratified splitting, cross-validation, the baseline model and the metrics
- TensorFlow / Keras: the CNN
- Matplotlib: saved plots
- pytest: automatic checks

**Data**
- [Kaggle: Kepler Labelled Time Series Data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)

**Not used in this version:** Lightkurve, Jupyter, the NASA Exoplanet Archive tables and the MAST archive.
```
why? Because we have used direct csv data from kaggle, which reduced our burden to manually sort the data from MAST or NASA Archive.
```

**Reference material**
- Shallue & Vanderburg, *Identifying Exoplanets with Deep Learning* (arXiv:1712.05044)
- [Google Research exoplanet-ml](https://github.com/google-research/exoplanet-ml) (original AstroNet; not redistributed here)
- [TensorFlow exoplanet CNN example](https://blog.tensorflow.org/2018/11/deep-learning-for-identifying-exoplanets.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

## 12. Data licensing and credit

The data is **not included** in this repository and stays under its own terms. Check the Kaggle dataset's license on its
page before reusing it, and credit it when you publish results. NASA and MAST data are publicly available, but check each
source's own acknowledgement requirements. Do the same for the exoplanet-ml repository. Do not commit large datasets or
third-party data packages here.

---

## 13. Limitations

- Only 37 labeled planets in the training file and 5 in the test file, so single numbers swing a lot. Use the
  cross-validation results as the main estimate.
- The labels are a convenience, not a complete catalogue of astrophysical truth, and categories like confirmed, candidate
  and false positive are collapsed into one binary label.
- No periods or epochs, so no folding, no global and local views, and no secondary-eclipse checks.
- Median-filter detrending can suppress very long transits.
- The CNN is small and was not tuned. Results come from a single seed.
- The Lightkurve milestone from the research guide was not completed (see section 9).
- Kepler and TESS data are not interchangeable without care (cadence, noise, systematics). Only Kepler-derived data was used.
- The model ranks signals. It does not detect dips and does not confirm planets.

---

## 14. What I learned

- **Detecting vs classifying:** finding a repeating dip is a different job from judging whether that dip looks like a planet.
- **Candidate vs confirmed:** a high model score is a ranking, not a confirmation. Confirmation needs follow-up observations
  and expert review.
- **Why not accuracy alone:** with so few planets, a model can be highly "accurate" while missing nearly all of them.
  PR-AUC, precision and recall tell the real story.
- **One split is not enough:** with so few planets, cross-validation showed how much the scores swing.
- **Choose the threshold on held-out predictions, never on the test set,** and use the test set once.
- **Errors teach the most:** plotting the confident false alarms and misses showed what fools the model.
- **Concepts from the research guide that this version does not implement:** folding a light curve on its period (it makes a
  faint repeating dip clear enough to learn from), building global and local views, and splitting views by star.
- **What I learned in Electronics & Communication, and where it shows up:**

| In this project | Electronics idea |
|---|---|
| Clipping upward spikes, robust scaling | Noise removal, outlier clipping |
| Median-filter detrending | Removing slow drift, similar to high-pass filtering |
| Scaling each star | Signal normalization |
| `Conv1D` layers in the CNN | Sliding-window filtering (convolution) with learned coefficients |
| Faint dip versus noise | SNR, which is the core challenge |
| *(not implemented)* Folding on the period | Averaging repeated cycles to lift a weak signal above noise |
| *(not implemented)* Global and local views | Resampling and binning to a fixed length |

---

## Documentation

- `docs/RESEARCH.md`: the research guide this project follows.
- `docs/01_project_flow.md`: the pipeline from raw data to the write-up.
- `docs/02_concepts.md`: every concept used, in plain words.
- `docs/03_outputs_explained.md`: what each run prints and what it means.
- `docs/04_code_reference.md`: each file and function, and its purpose.
- `docs/05_why_these_choices.md`: why each choice was made.
- `docs/06_final_stages.md`: cross-validation, error analysis, final test and ranked lists.
- `docs/steps/`: one short guide per step (setup, data, preprocessing, baseline, CNN, cross-validation, error analysis,
  final test, ranked list and summary).
