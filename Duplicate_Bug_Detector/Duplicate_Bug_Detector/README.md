# Duplicate Bug Detector

**Student:** Vasantha Perugu  
**Project:** Pre-save duplicate bug detection for GitHub issues

## Project idea

When a user is about to submit a new GitHub bug report, the same bug may already exist. This project checks the new report against existing GitHub issues and shows the closest possible duplicate matches before the report is saved.

## What the notebook does

- Collects recent GitHub issues from selected repositories
- Removes pull requests and cleans the issue dataset
- Uses both the bug title and bug description for matching
- Builds similarity features from title, description, combined text, and semantic text
- Creates candidate duplicate issue pairs for review
- Builds a labeled ground-truth evaluation file
- Tunes the duplicate-score threshold on validation data
- Reports precision, recall, F1-score, and confusion matrix on held-out test data
- Saves artifacts for the Streamlit demo app
- Builds a Streamlit app called **Pre-Save Duplicate Bug Detector**

## Repository structure

```text
.
├── Duplicate_Bug_Detector.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
├── models/
├── outputs/
│   ├── figures/
│   └── tables/
└── docs/
```

## Main notebook sections

1. Approach  
2. Data Settings  
3. GitHub Access  
4. Collect Recent GitHub Issues  
5. Clean the Dataset and Check the Shape  
6. Visual Read of the Dataset  
7. Prepare Title and Bug Description Text for Similarity Search  
8. Build Title, Description, and Semantic Features  
9. Find Similar Bug Reports Without Hanging  
10. Manual Review File  
11. Ground-Truth Pair Evaluation  
12. Test the Detector Inside the Notebook  
13. Save Artifacts for the App  
14. Build the Streamlit UI  
15. Open the Streamlit App  
16. Final Submission Notes

## How to run the notebook

1. Install Python 3.10 or newer.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Open `Duplicate_Bug_Detector.ipynb` in Jupyter Notebook or JupyterLab.
4. Create a GitHub token and either:
   - paste it when the notebook asks for it, or
   - set it as an environment variable named `GITHUB_TOKEN`.
5. Run the notebook from top to bottom.
6. After Section 13, the notebook saves the model artifact in `models/`.
7. After Section 14, the Streamlit app file is available in `app/streamlit_app.py`.

## How to run the app after the notebook finishes

```bash
streamlit run app/streamlit_app.py
```

The app needs this file, which is created by the notebook:

```text
models/duplicate_bug_detector_artifacts.joblib
```

If the app says the artifact is missing, run the notebook through Section 13 first.

## Files to upload to GitHub

Upload these files first:

- `Duplicate_Bug_Detector.ipynb`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `app/streamlit_app.py`
- folder placeholders such as `.gitkeep` files inside `data/`, `models/`, `outputs/`, and `docs/`

Do not upload your GitHub token. Do not hard-code the token in the notebook or app.

## Notes

- The project uses live GitHub issue data.
- Raw API pages and model artifacts can become large, so they are ignored by default in `.gitignore`.
- If your instructor wants generated CSVs, figures, or the model artifact, upload only the specific required output files after checking file size.
