import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Atlas Demo", layout="wide")
st.title("Atlas — Working Demo")
st.write("Type a business description below and see it move through each real, tested step.")

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

cobs, naics, rules = load_data()
vec, appetite_clf = train_appetite_classifier(naics)

with st.spinner("Loading semantic matching model (first run only, may take a minute)..."):
    model = load_semantic_model()
    cob_texts, cob_embeddings = encode_cobs(model, cobs)

cob_names = cobs["Hiscox_COB"].tolist()

def check_rules(name, rules_df):
    nl = name.lower()
    for _, r in rules_df.iterrows():
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

query = st.text_input("Enter a business description:", placeholder="e.g. Cardiologist office")

if query:
    st.divider()
    st.subheader("Step 0 — Appetite Classifier")
    X_query = vec.transform([query])
    pred = appetite_clf.predict(X_query)[0]
    prob = appetite_clf.predict_proba(X_query)[0]
    classes = list(appetite_clf.classes_)
    confidence = prob[classes.index(pred)]
    is_short = len(query.split()) <= 2

    st.write(f"**Prediction:** {'In-Appetite' if pred=='Yes' else 'Out of Appetite (OOA)'}  |  **Confidence:** {confidence:.2f}")
    if is_short:
        st.warning("Input is very short (1-2 words) — this classifier is known to be less reliable on short inputs. Treat this result with extra caution.")

    if pred == "No" and confidence > 0.75 and not is_short:
        st.error("Result: OOA — high confidence. Stopping here (matches Step 0 design).")
    else:
        st.subheader("Step 2 — Rules Filter")
        vague = check_vague_input(query)
        rule_hit = check_rules(query, rules)

        if vague:
            st.warning('Input starts with a generic catch-all term ("Other"/"Miscellaneous"/"NOC") — flagged for human review regardless of any match found below.')

        if rule_hit is not None:
            st.info(f"**Rule matched:** \"{rule_hit['Pattern_or_Phrase']}\" -> **{rule_hit['Direction']}** ({rule_hit['Rule_Type']})")
            st.write(f"Notes: {rule_hit['Notes']}")
        else:
            st.write("No rule matched — continuing to semantic matching.")

            st.subheader("Step 3 — Semantic Matching (against 543 COBs)")
            q_emb = model.encode([query], normalize_embeddings=True)
            sims = q_emb @ cob_embeddings.T
            top3_idx = np.argsort(-sims[0])[:3]

            for rank, idx in enumerate(top3_idx, start=1):
                st.write(f"**#{rank}: {cob_names[idx]}**  (similarity: {sims[0][idx]:.3f})")

            if sims[0][top3_idx[0]] < 0.45:
                st.warning("Top match confidence is low — this case likely needs human review rather than auto-accept.")
