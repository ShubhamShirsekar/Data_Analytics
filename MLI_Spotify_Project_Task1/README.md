# Spotify Family Account Playlist Reconstruction — Task 1

This repository contains the notebook, generated outputs, and reports for reconstructing deleted "top-of-the-year" playlists from a mixed family account. The analysis is implemented in `Spotify_project_Task1_v2.ipynb` and follows a reproducible EDA → feature engineering → modeling workflow.

Summary
- Goal: Predict missing `user` and `top_year` labels for unlabeled songs (100 unlabeled vs ~3,500 labeled).
- Strategy: Two-stage approach — global user classification, then user-specific year classification (5 separate year models).

Quick repository contents
- `Spotify_project_Task1_v2.ipynb` — primary notebook with full EDA, feature engineering, and model training.
- `mixed_playlist.csv` — input dataset (labeled + unlabeled songs).
- `reconstructed_playlists/` — generated per-cohort CSV files (alpha/beta/gamma/delta/epsilon by year).
- `unlabeled_predictions.csv` — final user/year predictions for unlabeled songs.
- `low_confidence_predictions.csv` — subset of low-confidence predictions flagged for manual review.
- `reconstruction_summary.csv` — summary of generated outputs and counts.
- `requirements.txt` — Python dependencies used to reproduce the analysis.
- Reports: `executive_summary.txt`, `final_project_report.txt`, `quality_validation_report.txt`.

How the notebook is organized (section-by-section)
- Section 1 — Imports and setup: standard scientific stack (pandas, numpy, scikit-learn, seaborn, matplotlib) and utility settings.
- Section 2 — Data loading & initial exploration: reads `mixed_playlist.csv`, drops NAs, prints types and basic stats.
- Section 3 — Identify labeled vs unlabeled: splits rows using markers (e.g., `deleted by hacker`) into `labeled_df` and `unlabeled_df`.
- Section 4 — Exploratory Data Analysis (EDA):
  - Distribution analysis for `user` and `top_year`.
  - Identification of continuous and categorical features; temporal trend analysis (per-feature correlation with year).
  - Categorical analyses (time_signature, key, mode) with chi-square tests.
  - Popularity analysis with ANOVA (popularity is a strong user predictor but a leakage feature for year).
  - Text analysis for `artist` and `album` (artist/album diversity, loyalty, temporal concentration).
  - Leakage detection: builds `song_id` and checks overlap between labeled and unlabeled sets; warns if leakage exists.
- Section 5 — Feature engineering & preparation:
  - Feature categories defined: continuous audio, categorical, text-derived, leakage features.
  - Missing value handling and one-hot encoding for categorical features.
  - Text-derived features: user-artist affinity counts, user-specific top-artist flags, artist temporal features (peak/first year), album behavior features, and global artist popularity.
  - Constructs final feature sets for (a) global user model (includes popularity) and (b) per-user year models (excludes popularity to avoid leakage).
  - Scaling: `StandardScaler` for continuous/text-count features (global scaler for user model; per-user scalers for year models).
- Section 6 — Model training: User classification
  - Global Random Forest classifier for user prediction (hyperparameters in notebook).
  - Validation split, cross-validation, confusion matrix, feature importance analysis.
- Section 7 — Model training: Year classification
  - Per-user Gradient Boosting models for year prediction (one model per user), with validation, CV, feature importance and adjacent-year confusion analysis.
- Final steps — Prediction & outputs
  - Sequence: predict `user` for unlabeled rows → route each row to the corresponding user-year model → predict `top_year`.
  - Save outputs to `unlabeled_predictions.csv`, `low_confidence_predictions.csv`, and `reconstructed_playlists/`.

Reproducibility (setup & run)
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the notebook interactively in JupyterLab/Notebook. 

Notes & best practices
- Popularity is a leakage feature for year prediction: exclude `popularity` when training year models.
- Text features: if leakage (song overlaps) is detected, avoid using exact song/name matches; use aggregated features only (artist popularity, user-artist affinity counts, etc.).
- The modeling pipeline is: user model (global) → user-specific year models (per user). This improves year prediction by conditioning on user-specific patterns.
- Verify outputs by checking `reconstruction_summary.csv` after running the notebook.
