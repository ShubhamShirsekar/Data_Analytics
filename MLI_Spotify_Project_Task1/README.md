# ML Spotify Project (Task 1)

This folder contains the Jupyter notebook and related outputs for the ML Spotify reconstruction project.

Quick notes to prepare for manual upload:

- The repository should exclude the virtual environment folder `.venv/` and Jupyter checkpoints — a `.gitignore` is included.
- To reproduce results locally, create a Python virtual environment and install dependencies from `requirements.txt`.

Running the notebook (Windows):

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# run the provided wrapper to execute the notebook on Windows
.\.venv\Scripts\python.exe run_nb.py
```

Files of interest:

- `Spotify_project_Task1_v2.ipynb` — main notebook (executed copy available)
- `executive_summary.txt`, `final_project_report.txt` — textual reports with metrics
- `reconstructed_playlists/` — generated CSVs
- `unlabeled_predictions.csv`, `low_confidence_predictions.csv`

If you want me to prepare a commit-ready branch instead, tell me and I will stage only the intended files.
