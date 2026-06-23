# Duplicate Bug Detector 

This folder contains the final notebook and supporting files for the Duplicate Bug Detector project.

## Main file

Open and run:

`vasantha_Duplicate_Bug_Detector.ipynb`

The notebook builds a duplicate-bug detector that compares a new bug report against existing GitHub issues using both the bug title and the bug description.

## What is included

- Final submission notebook
- Reproducible `requirements.txt`
- Streamlit app source in `app/streamlit_app.py`
- Project folder structure for data, models, outputs, and documentation
- Final demo PowerPoint in `docs/`

## How to run

1. Create or activate a Python environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Open the notebook and run cells from top to bottom.
4. The notebook uses cached data by default when available. If no cache exists, it collects public GitHub issues once and saves them for reuse.
5. Run the final Streamlit section to open the demo app.

## Reproducibility settings

The notebook is configured for stable grading/demo runs:

- `USE_LIVE_GITHUB_DATA = False`
- `USE_CACHE_IF_GITHUB_FAILS = True`
- `PREFER_SENTENCE_TRANSFORMER = False`
- LSA semantic matching is used by default to avoid fragile local transformer installation issues.

## Evaluation

The notebook evaluates the duplicate detector using a ground-truth pair file with:

`source_issue_key, target_issue_key, label, label_source`

It tunes the decision threshold on validation data and reports precision, recall, F1-score, and a confusion matrix on held-out test data.
