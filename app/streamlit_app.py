import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "duplicate_bug_detector_artifacts.joblib"
EXPECTED_APP_VERSION = "v10_original_collection_description_preview_fix"

st.set_page_config(page_title="Duplicate Bug Detector", layout="wide")
st.title("Pre-Save Duplicate Bug Detector")
st.write(
    "Enter a bug title and description before saving. The app compares both fields "
    "against the live-collected GitHub issue corpus and shows the closest duplicate-risk candidates."
)

if not ARTIFACT_PATH.exists():
    st.error("Model artifacts were not found. Run the notebook through Section 13 first.")
    st.stop()

artifacts = joblib.load(ARTIFACT_PATH)
required_keys = [
    "app_version",
    "body_vectorizer",
    "X_body",
    "thresholds",
    "evaluation_summary",
    "score_used_for_decision",
    "decision_logic",
]
missing_keys = [key for key in required_keys if key not in artifacts]
if missing_keys:
    st.error("The saved artifacts are from an older version. Run the notebook from Section 8 through Section 14, then restart the app.")
    st.write("Missing artifact keys:", missing_keys)
    st.stop()

app_version = artifacts.get("app_version", "unknown")
if app_version != EXPECTED_APP_VERSION:
    st.warning(f"Artifact version is {app_version}, expected {EXPECTED_APP_VERSION}. Re-run Sections 8-14 if results look stale.")

corpus = artifacts["corpus"].copy()
word_vectorizer = artifacts["word_vectorizer"]
char_vectorizer = artifacts["char_vectorizer"]
title_vectorizer = artifacts["title_vectorizer"]
body_vectorizer = artifacts["body_vectorizer"]
X_word = artifacts["X_word"]
X_char = artifacts["X_char"]
X_title = artifacts["X_title"]
X_body = artifacts["X_body"]
X_semantic = artifacts["X_semantic"]
weights = artifacts["weights"]
thresholds = artifacts["thresholds"]
evaluation_summary = artifacts["evaluation_summary"]
semantic_backend = artifacts["semantic_backend"]
semantic_model_name = artifacts["semantic_model_name"]
FIELD_MATCH_THRESHOLD = float(artifacts.get("field_match_threshold", 0.40))
WORD_MATCH_CAP = float(artifacts.get("word_match_cap", 0.40))
MIN_EXACT_DESCRIPTION_CHARS = int(artifacts.get("min_exact_description_chars", 30))
MIN_PARTIAL_DESCRIPTION_TOKENS = int(artifacts.get("min_partial_description_tokens", 12))
GENERIC_TITLE_WORDS = set(artifacts.get("generic_title_words", ["test", "bug", "issue", "error", "problem", "help"]))

lsa_model = artifacts.get("lsa_model")
semantic_model = None
if semantic_backend == "sentence_transformer":
    try:
        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer(semantic_model_name)
    except Exception as exc:
        st.error(
            "The artifacts were built with sentence-transformers, but this Streamlit environment cannot load that package. "
            "Re-run the notebook in this same environment so it can create the safe LSA fallback artifacts, then restart Streamlit."
        )
        st.caption(f"Loading problem: {exc!r}")
        st.stop()


def clean_issue_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^A-Za-z0-9_#./:+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def title_is_informative(clean_title):
    tokens = re.findall(r"[a-z0-9_#./:+\\-]+", clean_title.lower())
    meaningful_tokens = [token for token in tokens if token not in GENERIC_TITLE_WORDS and len(token) > 1]
    return len(meaningful_tokens) >= 2


def normalize_description_match_text(text):
    cleaned = clean_issue_text(text)
    return " ".join(re.findall(r"[a-z0-9_]+", cleaned.lower()))


def make_preview(text, max_chars=260):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def hybrid_score(semantic_score, body_score, title_score, word_score, char_score):
    word_score_used = min(float(word_score), WORD_MATCH_CAP)
    return float(
        weights["semantic"] * semantic_score
        + weights["body"] * body_score
        + weights["title"] * title_score
        + weights["word"] * word_score_used
        + weights["char"] * char_score
    )


def duplicate_evidence_level(semantic_score, body_score, title_score, word_score, char_score):
    if title_score >= 0.35 and body_score >= 0.20:
        return "title_and_description"
    if body_score >= 0.34 and (semantic_score >= 0.50 or word_score >= 0.22):
        return "description_dominant"
    if word_score >= 0.45 and body_score >= 0.15:
        return "combined_text"
    if semantic_score >= 0.72 and body_score >= 0.20:
        return "semantic_with_description"
    if title_score >= 0.75 and (body_score >= 0.10 or word_score >= 0.28 or char_score >= 0.35):
        return "strong_title_with_support"
    if title_score >= 0.75 or body_score >= 0.30:
        return "single_field_possible"
    if semantic_score >= 0.68 and (word_score >= 0.18 or char_score >= 0.22):
        return "semantic_possible"
    return "weak_or_generic"


def build_match_reason(semantic_score, body_score, title_score, word_score, char_score, final_score, duplicate_score, evidence_level):
    reasons = []
    if title_score >= 0.35:
        reasons.append("title similarity")
    if body_score >= 0.20:
        reasons.append("bug-description similarity")
    if word_score >= 0.22:
        reasons.append("shared technical words")
    if char_score >= 0.30:
        reasons.append("similar error/text fragments")
    if semantic_score >= 0.60:
        reasons.append("semantic meaning similarity")

    if evidence_level == "weak_or_generic":
        if semantic_score >= 0.60 and body_score < 0.20 and title_score < 0.35:
            return "Semantic similarity only; not enough title/description support for a duplicate"
        return "Weak or generic similarity; review only if context requires it"

    if evidence_level == "single_field_possible":
        return "Only one field is strongly similar; possible duplicate but not high confidence"

    return "Coherent evidence: " + ", ".join(reasons) if reasons else "Coherent hybrid title + description evidence"


def coherent_duplicate_score(semantic_score, body_score, title_score, word_score, char_score):
    final_score = hybrid_score(semantic_score, body_score, title_score, word_score, char_score)
    evidence_level = duplicate_evidence_level(semantic_score, body_score, title_score, word_score, char_score)

    if evidence_level in {"title_and_description", "description_dominant", "combined_text", "semantic_with_description"}:
        bonus = 0.04
        if title_score >= 0.50 and body_score >= 0.30:
            bonus += 0.05
        if word_score >= 0.40 or char_score >= 0.40:
            bonus += 0.03
        duplicate_score = min(1.0, final_score + bonus)
    elif evidence_level == "strong_title_with_support":
        duplicate_score = min(0.62, final_score + 0.04)
    elif evidence_level in {"single_field_possible", "semantic_possible"}:
        duplicate_score = min(0.52, max(final_score, 0.55 * title_score + 0.45 * body_score, 0.70 * body_score))
    else:
        duplicate_score = min(0.36, final_score)

    duplicate_score = float(np.clip(duplicate_score, 0.0, 1.0))
    reason = build_match_reason(
        semantic_score, body_score, title_score, word_score, char_score, final_score, duplicate_score, evidence_level
    )
    return final_score, duplicate_score, evidence_level, reason


def encode_semantic(title, body):
    query_text = f"{title}. {body}"[:2500]
    if semantic_backend == "sentence_transformer":
        return np.asarray(semantic_model.encode([query_text], normalize_embeddings=True), dtype=np.float32)

    clean_text = clean_issue_text(f"{title} {body}")
    return np.asarray(normalize(lsa_model.transform(word_vectorizer.transform([clean_text]))), dtype=np.float32)


def mixed_evidence_details(repo_indices, title_scores, body_scores, duplicate_scores):
    title_top_pos = int(np.argmax(title_scores))
    body_top_pos = int(np.argmax(body_scores))

    title_top_row = corpus.iloc[int(repo_indices[title_top_pos])]
    body_top_row = corpus.iloc[int(repo_indices[body_top_pos])]

    mixed = (
        title_top_row["issue_key"] != body_top_row["issue_key"]
        and float(title_scores[title_top_pos]) >= 0.55
        and float(body_scores[body_top_pos]) >= 0.25
        and float(np.max(duplicate_scores)) < thresholds["likely_duplicate"]
    )

    return bool(mixed), {
        "title_issue": title_top_row["issue_key"],
        "title_score": float(title_scores[title_top_pos]),
        "body_issue": body_top_row["issue_key"],
        "body_score": float(body_scores[body_top_pos]),
    }


def decision(results, mixed_evidence):
    if results.empty:
        return "NO MATCHING DUPLICATE FOUND IN THE COLLECTED CORPUS", "success"

    if bool(results["exact_description_match"].any()):
        return "DUPLICATE OF THIS BUG", "error"

    direct_matches = results[
        (results["title_score"] >= FIELD_MATCH_THRESHOLD)
        & (results["body_score"] >= FIELD_MATCH_THRESHOLD)
    ]
    if not direct_matches.empty:
        return "DUPLICATE OF THIS BUG", "error"

    return "POSSIBLE DUPLICATE OF THESE BUGS", "warning"


def search(repo_name, title, body, top_k=10):
    repo_indices = corpus.index[corpus["repo"].eq(repo_name)].to_numpy()
    clean_title = clean_issue_text(title)
    clean_body = clean_issue_text(body)
    search_text = f"{clean_title} {clean_body}".strip()

    if not search_text:
        return pd.DataFrame(), False, {}

    informative_title = title_is_informative(clean_title)
    q_sem = encode_semantic(title, body)
    q_word = word_vectorizer.transform([search_text])
    q_char = char_vectorizer.transform([search_text])
    q_title = title_vectorizer.transform([clean_title])
    q_body = body_vectorizer.transform([clean_body])

    semantic_scores = cosine_similarity(q_sem, X_semantic[repo_indices]).ravel()
    body_scores = cosine_similarity(q_body, X_body[repo_indices]).ravel()
    title_scores = cosine_similarity(q_title, X_title[repo_indices]).ravel()
    word_scores = cosine_similarity(q_word, X_word[repo_indices]).ravel()
    char_scores = cosine_similarity(q_char, X_char[repo_indices]).ravel()

    query_match_body = normalize_description_match_text(body)
    repo_match_bodies = (
        corpus.iloc[repo_indices]["body"]
        .fillna("")
        .astype(str)
        .map(normalize_description_match_text)
        .to_numpy()
    )

    exact_description_matches = np.zeros(len(repo_indices), dtype=bool)
    partial_description_matches = np.zeros(len(repo_indices), dtype=bool)

    query_token_count = len(query_match_body.split())
    if len(query_match_body) >= MIN_EXACT_DESCRIPTION_CHARS:
        exact_description_matches = repo_match_bodies == query_match_body

    if query_token_count >= MIN_PARTIAL_DESCRIPTION_TOKENS:
        partial_description_matches = np.asarray([
            bool(candidate_body)
            and not bool(exact_description_matches[pos])
            and min(query_token_count, len(candidate_body.split())) >= MIN_PARTIAL_DESCRIPTION_TOKENS
            and (
                query_match_body in candidate_body
                or candidate_body in query_match_body
            )
            for pos, candidate_body in enumerate(repo_match_bodies)
        ], dtype=bool)

    description_text_matches = exact_description_matches | partial_description_matches
    body_scores = np.where(description_text_matches, 1.0, body_scores)

    final_scores = []
    duplicate_scores = []
    evidence_levels = []
    reasons = []
    for sem, body_score, title_score, word_score, char_score in zip(
        semantic_scores, body_scores, title_scores, word_scores, char_scores
    ):
        final, duplicate, evidence_level, reason = coherent_duplicate_score(
            float(sem), float(body_score), float(title_score), float(word_score), float(char_score)
        )
        final_scores.append(final)
        duplicate_scores.append(duplicate)
        evidence_levels.append(evidence_level)
        reasons.append(reason)

    final_scores = np.asarray(final_scores)
    duplicate_scores = np.asarray(duplicate_scores)
    mixed_evidence, mixed_details = mixed_evidence_details(repo_indices, title_scores, body_scores, duplicate_scores)

    ranked_positions = sorted(
        range(len(repo_indices)),
        key=lambda pos: (
            bool(description_text_matches[pos]),
            bool(exact_description_matches[pos]),
            float(duplicate_scores[pos]),
            float(body_scores[pos]),
            float(title_scores[pos]),
        ),
        reverse=True,
    )

    rows = []
    for pos in ranked_positions:
        description_pass = float(body_scores[pos]) >= FIELD_MATCH_THRESHOLD
        title_pass = informative_title and float(title_scores[pos]) >= FIELD_MATCH_THRESHOLD

        if not description_text_matches[pos] and not description_pass and not title_pass:
            continue

        row = corpus.iloc[int(repo_indices[pos])]
        rows.append({
            "rank": len(rows) + 1,
            "issue_key": row["issue_key"],
            "matched_title": row["title"],
            "matched_description_preview": make_preview(row["body"]),
            "duplicate_score": round(float(duplicate_scores[pos]), 4),
            "final_score": round(float(final_scores[pos]), 4),
            "semantic_score": round(float(semantic_scores[pos]), 4),
            "body_score": round(float(body_scores[pos]), 4),
            "title_score": round(float(title_scores[pos]), 4),
            "title_match_percent": round(float(title_scores[pos]) * 100, 1),
            "description_match_percent": round(float(body_scores[pos]) * 100, 1),
            "exact_description_match": bool(exact_description_matches[pos]),
            "partial_description_match": bool(partial_description_matches[pos]),
            "description_text_match": bool(description_text_matches[pos]),
            "description_match_type": (
                "exact_full_description"
                if exact_description_matches[pos]
                else "copied_preview_or_substring"
                if partial_description_matches[pos]
                else "similarity_score"
            ),
            "word_score": round(float(word_scores[pos]), 4),
            "word_score_used": round(min(float(word_scores[pos]), WORD_MATCH_CAP), 4),
            "char_score": round(float(char_scores[pos]), 4),
            "evidence_level": evidence_levels[pos],
            "match_reason": reasons[pos],
            "state": row["state"],
            "url": row["html_url"],
        })
        if len(rows) >= top_k:
            break

    return pd.DataFrame(rows), mixed_evidence, mixed_details


with st.sidebar:
    st.subheader("Model metadata")
    st.caption(f"App/artifact version: {app_version}")
    st.caption(f"Corpus size: {len(corpus):,}")
    st.caption("Decision score: duplicate_score")
    st.caption(f"Embedding model: {semantic_backend}")
    st.caption(f"Data mode: {artifacts.get('data_source_mode', 'not recorded')}")

    st.subheader("Held-out test evaluation")
    st.metric("Precision", f"{evaluation_summary.get('precision', 0):.3f}")
    st.metric("Recall", f"{evaluation_summary.get('recall', 0):.3f}")
    st.metric("F1-score", f"{evaluation_summary.get('f1', 0):.3f}")
    st.caption(f"Evaluation source: {evaluation_summary.get('evaluation_source', 'not available')}")
    st.caption(f"Metrics reported on: {evaluation_summary.get('metrics_reported_on', 'held-out test split')}")
    st.caption(f"Likely duplicate threshold: {thresholds['likely_duplicate']:.2f}")
    st.caption(f"Possible duplicate threshold: {thresholds['possible_duplicate']:.2f}")
    st.caption("Display rule: description ≥ 40%, or an informative title ≥ 40%")
    st.caption("Full descriptions and substantial copied notebook previews are checked before TF-IDF ranking")

repos = sorted(corpus["repo"].unique())
repo_name = st.selectbox("Repository", repos)

with st.expander("Load a real issue as a demo test"):
    demo_rows = corpus[corpus["repo"].eq(repo_name)].head(100).copy()
    demo_options = ["-- choose a demo issue --"] + [f"{r.issue_key} | {r.title}" for r in demo_rows.itertuples()]
    selected_demo = st.selectbox("Demo issue", demo_options)

    if selected_demo != "-- choose a demo issue --":
        issue_key = selected_demo.split(" | ", 1)[0]
        demo = corpus.loc[corpus["issue_key"].eq(issue_key)].iloc[0]
        st.session_state["bug_title"] = demo["title"]
        st.session_state["bug_body"] = demo["body"]
        st.caption("The demo issue is not excluded. A pre-save detector should find an existing issue if the new report repeats it.")

bug_title = st.text_input("Bug title", value=st.session_state.get("bug_title", ""))
bug_body = st.text_area("Bug description", value=st.session_state.get("bug_body", ""), height=180)
top_k = st.slider("Number of candidates", min_value=5, max_value=25, value=10, step=5)

if st.button("Check before save", type="primary"):
    if not clean_issue_text(f"{bug_title} {bug_body}"):
        st.warning("Enter a bug title or bug description first.")
        st.stop()

    results, mixed_evidence, mixed_details = search(repo_name, bug_title, bug_body, top_k=top_k)
    message, level = decision(results, mixed_evidence)

    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.success(message)

    if results.empty:
        st.info(
            "No title or description match reached 40% in the collected corpus. "
            "The app also checked for a complete copied description and a substantial copied notebook preview."
        )
    else:
        exact_matches = results[results["exact_description_match"]]
        direct_matches = results[
            results["exact_description_match"]
            | (
                (results["title_score"] >= FIELD_MATCH_THRESHOLD)
                & (results["body_score"] >= FIELD_MATCH_THRESHOLD)
            )
        ]

        if not direct_matches.empty:
            best = direct_matches.iloc[0]
            st.subheader("Duplicate of this bug")
            st.write(f"**Bug ID:** {best['issue_key']}")
            st.write(f"**Title:** {best['matched_title']}")
            st.write(f"**Description:** {best['matched_description_preview']}")
            st.write(
                f"**Title match:** {best['title_match_percent']:.1f}%  |  "
                f"**Description match:** {best['description_match_percent']:.1f}%"
            )
            if bool(best["exact_description_match"]):
                st.caption("The normalized full description exactly matches an issue in the collected corpus.")
            st.markdown(f"[Open the existing GitHub issue]({best['url']})")
        else:
            st.subheader("Possible duplicate of these bugs")
            for row in results.itertuples(index=False):
                st.markdown(f"### {row.issue_key}")
                st.write(f"**Title:** {row.matched_title}")
                st.write(f"**Description:** {row.matched_description_preview}")
                st.write(
                    f"**Title match:** {row.title_match_percent:.1f}%  |  "
                    f"**Description match:** {row.description_match_percent:.1f}%"
                )
                if bool(row.partial_description_match):
                    st.caption("The pasted text matches a substantial copied section or notebook preview of this issue.")
                st.markdown(f"[Open this GitHub issue]({row.url})")
                st.divider()

        with st.expander("Show technical similarity scores"):
            st.dataframe(results, use_container_width=True, hide_index=True)

    st.caption(
        "A generic one-word title such as 'test' does not create a title-only duplicate warning. "
        "Description matching is checked independently. Full descriptions and substantial copied notebook previews "
        "are detected before TF-IDF ranking."
    )
    st.caption("The app is a review assistant. It does not automatically reject a bug report.")
