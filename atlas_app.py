import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Atlas Batch Scoring", layout="wide")
st.title("Atlas — Batch Test & Scoring")
st.write("Upload partner names, run them all through Atlas, then download results to grade in Excel.")

@st.cache_resource
def load_data():
    cobs = pd.read_csv("atlas_cobs.csv")
    naics = pd.read_csv("atlas_naics.csv")
    rules = pd.read_csv("atlas_rules.csv")
    return cobs, naics, rules

@st.cache_resource
def train_appetite_classifier(_naics):
    texts = _naics["NAICS_Description"].dropna().tolist()
    labels = _naics.loc[_naics["NAICS_Description"].notna(), "In_Appetite"].tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, labels)
    return vec, clf

@st.cache_resource
def load_semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def encode_cobs(_model, _cobs):
    texts = (_cobs["Hiscox_COB"].fillna("") + ". " + _cobs["Definition"].fillna("")).tolist()
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return texts, embeddings

@st.cache_resource
def encode_naics(_model, _naics):
    df = _naics.dropna(subset=["NAICS_Description"]).reset_index(drop=True)
    texts = df["NAICS_Description"].tolist()
    embeddings = _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return df, embeddings

cobs, naics, rules = load_data()
vec, appetite_clf = train_appetite_classifier(naics)

with st.spinner("Loading models (first run only, may take a minute)..."):
    model = load_semantic_model()
    cob_texts, cob_embeddings = encode_cobs(model, cobs)
    naics_df, naics_embeddings = encode_naics(model, naics)

cob_names = cobs["Hiscox_COB"].tolist()

def check_rules(name, rules_df):
    nl = name.lower()
    for _, r in rules_df.iterrows():
        if r["Rule_Type"] != "sector_carve_in":
            continue
        if pd.isna(r["Pattern_or_Phrase"]) or not r["Pattern_or_Phrase"]:
            continue
        pat = str(r["Pattern_or_Phrase"]).lower()
        alts = [a.strip() for a in re.split(r"[/]", pat)]
        for alt in alts:
            if alt and alt in nl:
                return r
    for _, r in rules_df.iterrows():
        if r["Rule_Type"] == "sector_carve_in":
            continue
        if pd.isna(r["Pattern_or_Phrase"]) or not r["Pattern_or_Phrase"]:
            continue
        pat = str(r["Pattern_or_Phrase"]).lower()
        alts = [a.strip() for a in re.split(r"[/]", pat)]
        for alt in alts:
            if alt and alt in nl:
                return r
    return None

def check_vague_input(name):
    vague_prefixes = ["other ", "miscellaneous ", "noc "]
    return any(name.lower().strip().startswith(p) for p in vague_prefixes)

def run_pipeline(query):
    row = {"Partner_Name": query}
    X_query = vec.transform([query])
    pred = appetite_clf.predict(X_query)[0]
    prob = appetite_clf.predict_proba(X_query)[0]
    classes = list(appetite_clf.classes_)
    confidence = prob[classes.index(pred)]
    is_short = len(query.split()) <= 2

    row["Step0_Prediction"] = "In-Appetite" if pred == "Yes" else "OOA"
    row["Step0_Confidence"] = round(float(confidence), 3)
    row["Step0_Short_Input"] = is_short
    row["Step0_Override"] = ""

    if pred == "No" and confidence >= 0.80:
        q_emb_check = model.encode([query], normalize_embeddings=True)
        sims_check = q_emb_check @ cob_embeddings.T
        best_idx = int(np.argmax(sims_check[0]))
        best_score = float(sims_check[0][best_idx])

        if best_score < 0.75:
            row["Final_Stage"] = "Step0_OOA_Stop"
            row["Recommended_COB"] = "OOA"
            row["Rule_Hit"] = ""
            row["Top1_COB"] = "OOA"
            row["Top1_Score"] = ""
            row["Top2_COB"] = ""
            row["Top3_COB"] = ""
            row["NAICS_Closest_Desc"] = ""
            row["NAICS_COB"] = ""
            row["NAICS_Score"] = ""
            row["Disagreement_Category"] = ""
            return row
        else:
            row["Step0_Override"] = f"Overridden - near-exact match to {cob_names[best_idx]} ({best_score:.2f})"

    vague = check_vague_input(query)
    rule_hit = check_rules(query, rules)

    if rule_hit is not None and rule_hit["Rule_Type"] == "phrase_exclusion" and rule_hit["Direction"] == "OOA":
        row["Final_Stage"] = "Rule_OOA_Stop"
        row["Recommended_COB"] = "OOA"
        row["Rule_Hit"] = rule_hit["Pattern_or_Phrase"]
        row["Top1_COB"] = "OOA"
        row["Top1_Score"] = ""
        row["Top2_COB"] = ""
        row["Top3_COB"] = ""
        row["NAICS_Closest_Desc"] = ""
        row["NAICS_COB"] = ""
        row["NAICS_Score"] = ""
        row["Disagreement_Category"] = ""
        return row

    row["Rule_Hit"] = rule_hit["Pattern_or_Phrase"] if rule_hit is not None else ""
    row["Vague_Input_Flag"] = vague

    q_emb = model.encode([query], normalize_embeddings=True)
    sims = q_emb @ cob_embeddings.T
    top3_idx = np.argsort(-sims[0])[:3]
    top_cob = cob_names[top3_idx[0]]
    top_score = float(sims[0][top3_idx[0]])

    row["Final_Stage"] = "Semantic_Match"
    row["Top1_COB"] = top_cob
    row["Top1_Score"] = round(top_score, 3)
    row["Top2_COB"] = f"{cob_names[top3_idx[1]]} ({sims[0][top3_idx[1]]:.3f})"
    row["Top3_COB"] = f"{cob_names[top3_idx[2]]} ({sims[0][top3_idx[2]]:.3f})"

    naics_sims = q_emb @ naics_embeddings.T
    best_naics_idx = int(np.argmax(naics_sims[0]))
    best_naics_score = float(naics_sims[0][best_naics_idx])
    naics_row = naics_df.iloc[best_naics_idx]
    naics_cob = naics_row["Hiscox_COB"]

    row["NAICS_Closest_Desc"] = naics_row["NAICS_Description"]
    row["NAICS_COB"] = naics_cob
    row["NAICS_Score"] = round(best_naics_score, 3)

    if naics_cob == top_cob:
        row["Disagreement_Category"] = "Agree"
        row["Recommended_COB"] = top_cob
    elif best_naics_score >= 0.85:
        row["Disagreement_Category"] = "Strong_Disagree_Trust_NAICS"
        row["Recommended_COB"] = naics_cob
    elif top_score >= 0.70 and best_naics_score < 0.70:
        row["Disagreement_Category"] = "Mild_Disagree_Trust_Semantic"
        row["Recommended_COB"] = top_cob
    else:
        row["Disagreement_Category"] = "Genuine_Disagree_Toss_Up"
        row["Recommended_COB"] = f"UNCLEAR - review needed ({top_cob} vs {naics_cob})"

    return row

st.subheader("Step 1: Upload partner names")
uploaded_file = st.file_uploader("Upload a CSV or Excel file with partner names (one column)", type=["csv", "xlsx"])

names = []
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        upload_df = pd.read_csv(uploaded_file)
    else:
        upload_df = pd.read_excel(uploaded_file)

    st.write("Preview:")
    st.dataframe(upload_df.head())

    col_choice = st.selectbox("Which column has the partner names?", upload_df.columns.tolist())
    names = upload_df[col_choice].dropna().astype(str).tolist()
    st.write(f"Found {len(names)} names.")

if st.button("Run batch") and names:
    if len(names) > 300:
        st.warning(f"You have {len(names)} names - only processing the first 300 to keep this responsive.")
        names = names[:300]

    progress = st.progress(0)
    results = []
    for i, name in enumerate(names):
        results.append(run_pipeline(name))
        progress.progress((i + 1) / len(names))

    df = pd.DataFrame(results)

    cols = df.columns.tolist()
    cols.remove("Recommended_COB")
    cols.insert(1, "Recommended_COB")
    df = df[cols]

    st.session_state["batch_results"] = df

if "batch_results" in st.session_state:
    st.subheader("Step 2: Download and grade in Excel")
    st.write("Open this in Excel, add a grading column with dropdown validation, and review at your own pace.")
    st.dataframe(st.session_state["batch_results"].head(20))
    csv_out = st.session_state["batch_results"].to_csv(index=False)
    st.download_button("Download results as CSV", csv_out, "atlas_batch_results.csv", "text/csv")
