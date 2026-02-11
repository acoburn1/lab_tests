# Model Training Repository for Category Learning

Purpose: Utilities, training code, and analysis scripts for producing and visualizing neural-network category learning data

Branches
- `master` — main branch, for running locally
- `openhpc` — different `program.py` + additional bash/slurm scripts for running on openhpc
Repository layout (key files / directories)
- `program.py` — main orchestration for data prep, training calls, and analysis runs
- `configs/` — preset .yaml files for configuring data, model specs, and (soon) output
- `Output/` — plotting and output helpers:
  - `MatrixOutput.py` — matrix plotting utilities
  - `StatOutput.py`, `PCAOutput.py` — statistical and PCA plotting utilities
- `Model/` — model implementations (training/evaluation)
- `Prep/` — data loading and preprocessing utilities
- `Results/Data/` — per-run `.npz` result files (e.g. `p_m1.npz`)
- `Results/Analysis/` — generated plots and derived outputs
- `Data/` — input CSV files and reference matrices

Data / `.npz` format expectations
- Each run saves a mapping of result arrays keyed by names (e.g. `hidden_matrices`).
- `hidden_matrices` is expected to be an array-like sequence indexed by epoch where each element is a numeric 2D ndarray representing activity/co-occurrence for that epoch.
- For best performance and safety, `hidden_matrices` should be saved as a numeric NumPy array (not `dtype=object`) so downstream code can stack and average without Python-level objects.
 listing bad files if no valid matrices remain.

