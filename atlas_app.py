import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Atlas", layout="wide")
st.title("Atlas — Class of Business Matcher")

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

STEP0_STOP_THRESHOLD = 0.80
OVERRIDE_NAME_MATCH_THRESHOLD = 0.75
SEMANTIC_OWN_CONFIDENCE_THRESHOLD = 0.60
WEAK_WINNER_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.45

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
    confidence = float(prob[classes.index(pred)])
    is_short = len(query.split()) <= 2

    row["Step0_Prediction"] = "In-Appetite" if pred == "Yes" else "OOA"
    row["Step0_Confidence"] = round(confidence, 3)
    row["Step0_Short_Input"] = is_short
    row["Step0_Override"] = ""

    def empty_match_fields():
        row["Rule_Hit"] = ""
        row["Top1_COB"] = "OOA"
        row["Top1_Score"] = ""
        row["Top2_COB"] = ""
        row["Top3_COB"] = ""
        row["NAICS_Closest_Desc"] = ""
        row["NAICS_COB"] = ""
        row["NAICS_Score"] = ""
        row["Disagreement_Category"] = ""
        row["Needs_Review"] = False
        row["Review_Confidence"] = "High - confident OOA stop"

    if pred == "No" and confidence >= STEP0_STOP_THRESHOLD:
        q_emb_check = model.encode([query], normalize_embeddings=True)
        sims_check = q_emb_check @ cob_embeddings.T
        best_idx = int(np.argmax(sims_check[0]))
        best_score = float(sims_check[0][best_idx])

        if best_score < OVERRIDE_NAME_MATCH_THRESHOLD:
            row["Final_Stage"] = "Step0_OOA_Stop"
            row["Recommended_COB"] = "OOA"
            empty_match_fields()
            return row
        else:
            row["Step0_Override"] = f"Overridden - near-exact match to {cob_names[best_idx]} ({best_score:.2f})"

    vague = check_vague_input(query)
    rule_hit = check_rules(query, rules)
    row["Vague_Input_Flag"] = vague

    if rule_hit is not None and rule_hit["Rule_Type"] == "phrase_exclusion" and rule_hit["Direction"] == "OOA":
        row["Final_Stage"] = "Rule_OOA_Stop"
        row["Recommended_COB"] = "OOA"
        empty_match_fields()
        row["Rule_Hit"] = rule_hit["Pattern_or_Phrase"]
        return row

    row["Rule_Hit"] = rule_hit["Pattern_or_Phrase"] if rule_hit is not None else ""

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
        row["Needs_Review"] = False
        row["Review_Confidence"] = "High - both methods agree"
    else:
        row["Needs_Review"] = True
        if top_score >= SEMANTIC_OWN_CONFIDENCE_THRESHOLD:
            row["Disagreement_Category"] = "Trust_Semantic"
            row["Recommended_COB"] = top_cob
            winning_score = top_score
        else:
            row["Disagreement_Category"] = "Trust_NAICS"
            row["Recommended_COB"] = naics_cob
            winning_score = best_naics_score

        if winning_score >= WEAK_WINNER_THRESHOLD:
            row["Review_Confidence"] = f"Quick confirm - recommendation likely correct (~87% historically), winning score {winning_score:.2f}"
        else:
            row["Review_Confidence"] = f"Genuinely unclear - even the winning answer scored weak ({winning_score:.2f}) - full manual review needed"

    return row

mode = st.radio("Mode:", ["Single lookup", "Batch (upload a file)"], horizontal=True)
st.divider()

if mode == "Single lookup":
    query = st.text_input("Enter a business description:", placeholder="e.g. Cardiologist office")

    if query:
        result = run_pipeline(query)

        st.subheader("Step 0 — Appetite Classifier")
        st.write(f"**Prediction:** {result['Step0_Prediction']}  |  **Confidence:** {result['Step0_Confidence']:.2f}")
        if result["Step0_Short_Input"] and result["Step0_Confidence"] < STEP0_STOP_THRESHOLD:
            st.warning("Input is short and confidence is borderline — treating with extra caution.")
        if result["Step0_Override"]:
            st.warning(f"Step 0 said OOA with high confidence, but overridden: {result['Step0_Override']}")

        if result["Final_Stage"] == "Step0_OOA_Stop":
            st.error("Result: OOA — high confidence, no near-exact COB name match found. Stopping here.")
        elif result["Final_Stage"] == "Rule_OOA_Stop":
            st.error(f"Result: OOA — rule matched (\"{result['Rule_Hit']}\"). Stopping here.")
        else:
            st.subheader("Step 2 — Rules Filter")
            if result["Vague_Input_Flag"]:
                st.warning('Input starts with a generic catch-all term ("Other"/"Miscellaneous"/"NOC") — flagged for human review.')
            if result["Rule_Hit"]:
                st.info(f"Carve-in rule matched: \"{result['Rule_Hit']}\" — confirms In-Appetite, continuing to find specific COB.")
            else:
                st.write("No rule matched — continuing to semantic matching.")

            st.subheader("Step 3 — Semantic Matching (against 543 COBs)")
            st.write(f"**#1: {result['Top1_COB']}**  (similarity: {result['Top1_Score']})")
            st.write(f"#2: {result['Top2_COB']}")
            st.write(f"#3: {result['Top3_COB']}")
            if result["Top1_Score"] and result["Top1_Score"] < LOW_CONFIDENCE_THRESHOLD:
                st.warning("Top match confidence is low — likely needs human review.")

            st.subheader("Step 4 — NAICS Secondary Check")
            st.write(f"Closest NAICS description: *\"{result['NAICS_Closest_Desc']}\"* (similarity: {result['NAICS_Score']})")
            st.write(f"That row's known outcome: **{result['NAICS_COB']}**")

            if not result["Needs_Review"]:
                st.success(f"Agreement: both methods point to the same answer. **Recommended COB: {result['Recommended_COB']}**")
            elif result["Disagreement_Category"] == "Trust_Semantic":
                if "Quick confirm" in result["Review_Confidence"]:
                    st.info(f"🔵 FLAGGED FOR REVIEW (quick confirm) — semantic matching scored confidently on its own ({result['Top1_Score']:.2f}). **Recommended COB: {result['Recommended_COB']}**")
                else:
                    st.warning(f"🔴 FLAGGED FOR REVIEW (genuinely unclear) — {result['Review_Confidence']}. **Best guess: {result['Recommended_COB']}**")
            else:
                if "Quick confirm" in result["Review_Confidence"]:
                    st.info(f"🔵 FLAGGED FOR REVIEW (quick confirm) — semantic was weak, trusting NAICS. **Recommended COB: {result['Recommended_COB']}**")
                else:
                    st.warning(f"🔴 FLAGGED FOR REVIEW (genuinely unclear) — {result['Review_Confidence']}. **Best guess: {result['Recommended_COB']}**")

else:
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
        priority_cols = ["Partner_Name", "Recommended_COB", "Needs_Review", "Review_Confidence"]
        remaining_cols = [c for c in df.columns if c not in priority_cols]
        df = df[priority_cols + remaining_cols]

        st.session_state["batch_results"] = df

    if "batch_results" in st.session_state:
        st.subheader("Step 2: Download and grade in Excel")
        st.dataframe(st.session_state["batch_results"].head(20))
        csv_out = st.session_state["batch_results"].to_csv(index=False)
        st.download_button("Download results as CSV", csv_out, "atlas_batch_results.csv", "text/csv")
