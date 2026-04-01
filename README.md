# NASA Exoplanet Analysis

A machine learning project to classify Kepler exoplanet candidates using NASA's public dataset and hunt for undiscovered planets! 

---

## Project Overview

The [Kepler Space Telescope](https://www.nasa.gov/mission_pages/kepler/main/index.html) monitored over 150,000 stars to detect exoplanets via the **transit method** measuring the tiny dip in a star's brightness as a planet passes in front of it.

NASA's dataset contains thousands of **Kepler Objects of Interest (KOIs)**, each labeled as:
-  `CONFIRMED`  a verified exoplanet
-  `FALSE POSITIVE`  a signal that turned out not to be a planet
-  `CANDIDATE`  unverified signals awaiting confirmation

The goal of this project is to **train a classifier on confirmed and false positive signals**, then use it to **predict which candidates are most likely to be real exoplanets**.

---

##  Dataset

**Source:** [NASA Exoplanet Archive - Cumulative KOI Table](https://exoplanetarchive.ipac.caltech.edu/)

| Property | Value |
|---|---|
| Total objects | ~9,500 KOIs |
| Training set | CONFIRMED + FALSE POSITIVE (~7,500 rows) |
| Prediction set | CANDIDATE (~1,900 rows) |
| Features used | 11 physical measurements |

### Key Features

| Feature | Physical meaning | Unit |
|---|---|---|
| `koi_prad` | Planet radius | Earth radii R⊕ |
| `koi_model_snr` | Signal-to-noise ratio of the transit model | - |
| `koi_impact` | Impact parameter (transit geometry) |-|
| `koi_period` | Orbital period | days |
| `koi_duration` | Transit duration | hours |
| `koi_depth` | Transit depth (brightness drop) | ppm |
| `koi_insol` | Insolation flux received (1 = Earth) | F⊕ |
| `koi_teq` | Equilibrium temperature | Kelvin |
| `koi_steff` | Host star effective temperature | Kelvin |
| `koi_slogg` | Stellar surface gravity | log g |
| `koi_srad` | Stellar radius | Solar radii R☉ |

>  **Note on data leakage:** `koi_score` (NASA confidence score) and `koi_fpflag_*` (false positive flags) were deliberately excluded - they are derived from the disposition itself and would give the model an unfair advantage.

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution of dispositions (CONFIRMED / FALSE POSITIVE / CANDIDATE)
- Feature distributions and outlier detection
- Correlation analysis between physical features
- False positive flag analysis (99.99% of FALSE POSITIVES had at least one flag set → confirmed leakage)

### 2. Preprocessing
- Removed ~3% of rows with missing values (`dropna`)
- Encoded target: `CONFIRMED = 1`, `FALSE POSITIVE = 0`
- Stratified train/test split (80/20) to preserve class balance

### 3. Model Training
- **Algorithm:** Random Forest Classifier
- **Parameters:** 100 estimators, `class_weight='balanced'`, `random_state=42`
- **Stratified split** to maintain class proportions across train and test sets

### 4. Prediction on Candidates
- Applied the trained model to all 1,979 CANDIDATE signals
- Used `predict_proba()` to obtain a confidence score for each candidate
- Ranked candidates by probability of being a confirmed exoplanet

### 5. Validation
- Cross-referenced top predictions with the **NASA Exoplanet Archive API**
- Queried confirmed planets to check if any candidates had been confirmed since the dataset was created

---

## Results

### Model Performance

| Metric | FALSE POSITIVE (0) | CONFIRMED (1) | Overall |
|---|---|---|---|
| Precision | 0.93 | 0.87 | 0.91 |
| Recall | 0.92 | 0.89 | 0.91 |
| F1-score | 0.93 | 0.88 | 0.91 |
| Accuracy | - | - | **91%** |

### Feature Importance

The most predictive physical features were:

1. `koi_prad` (0.20) - Planet radius
2. `koi_model_snr` (0.15) - Signal-to-noise ratio
3. `koi_impact` (0.11) - Transit impact parameter
4. `koi_period` (0.11) - Orbital period

### Candidate Predictions

- **36 candidates** were classified with a probability > 90% of being a real exoplanet
- None of these 36 have been confirmed by NASA in the current archive, they remain **promising unverified signals** 
- All top candidates are still labeled `CANDIDATE` in the original NASA dataset, suggesting they await further observation

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly requests
```

### Run the notebook

```bash
git clone https://github.com/Soupirr/nasa-exoplanet-analysis.git
cd nasa-exoplanet-analysis
jupyter notebook
```

Then open `Cumulative_ML.ipynb` and run all cells.


---

## Author

**Soupirr** - [github.com/Soupirr](https://github.com/Soupirr)

---

### License

This project is open source and available under the [MIT License](LICENSE).

---

*Data courtesy of the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration.*
