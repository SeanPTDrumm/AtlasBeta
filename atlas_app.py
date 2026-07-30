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

@st.cache_resource
def encode_naics(_model, _naics):
    df = _naics.dropna(subset=["NAICS_Description"]).reset_index(drop=True)
    texts = df["NAICS_Description"].tolist()
    embeddings = _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return df, embeddings

cobs, naics, rules = load_data()
vec, appetite_clf = train_appetite_classifier(naics)

with st.spinner("Loading semantic matching model (first run only, may take a minute)..."):
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

    borderline = confidence < 0.80

    if is_short and borderline:
        st.warning("Input is short AND confidence is borderline — this is exactly the pattern that caused a real error in testing ('Plumbers' misfired). Treating this result with extra caution and continuing to further checks.")
    elif is_short:
        st.caption("Input is short, but confidence is high enough that this result is likely still reliable.")

    if pred == "No" and confidence >= 0.80:
        # Before trusting a confident OOA call, do a cheap safety check:
        # does this text nearly-exactly match a real COB name? If so, that's
        # a stronger signal than the classifier and should override it.
        q_emb_check = model.encode([query], normalize_embeddings=True)
        sims_check = q_emb_check @ cob_embeddings.T
        best_idx = int(np.argmax(sims_check[0]))
        best_score = float(sims_check[0][best_idx])

        if best_score >= 0.75:
            st.warning(f"Step 0 said OOA with high confidence, BUT this text nearly matches an actual COB name: \"{cob_names[best_idx]}\" (similarity {best_score:.2f}). Overriding the OOA stop — this is a stronger signal. Continuing to further checks.")
            override_ooa_stop = True
        else:
            st.error("Result: OOA — high confidence. Stopping here (matches Step 0 design).")
            override_ooa_stop = False
    else:
        override_ooa_stop = True

    if override_ooa_stop:
        st.subheader("Step 2 — Rules Filter")
        vague = check_vague_input(query)
        rule_hit = check_rules(query, rules)

        if vague:
            st.warning('Input starts with a generic catch-all term ("Other"/"Miscellaneous"/"NOC") — flagged for human review regardless of any match found below.')

        stop_here = False
        if rule_hit is not None:
            st.info(f"**Rule matched:** \"{rule_hit['Pattern_or_Phrase']}\" -> **{rule_hit['Direction']}** ({rule_hit['Rule_Type']})")
            st.write(f"Notes: {rule_hit['Notes']}")
            if rule_hit['Rule_Type'] == 'phrase_exclusion' and rule_hit['Direction'] == 'OOA':
                st.error("This is a complete answer (OOA exclusion rule) — stopping here.")
                stop_here = True
            elif rule_hit['Rule_Type'] == 'sector_carve_in':
                st.write("This rule only confirms the case is In-Appetite (overriding a sector-level OOA pattern) — it does NOT specify which COB. Continuing to semantic matching to find the specific COB.")
        else:
            st.write("No rule matched — continuing to semantic matching.")

        if not stop_here:
            st.subheader("Step 3 — Semantic Matching (against 543 COBs)")
            q_emb = model.encode([query], normalize_embeddings=True)
            sims = q_emb @ cob_embeddings.T
            top3_idx = np.argsort(-sims[0])[:3]

            for rank, idx in enumerate(top3_idx, start=1):
                st.write(f"**#{rank}: {cob_names[idx]}**  (similarity: {sims[0][idx]:.3f})")

            top_cob = cob_names[top3_idx[0]]
            top_cob_score = sims[0][top3_idx[0]]

            if top_cob_score < 0.45:
                st.warning("Top match confidence is low — this case likely needs human review rather than auto-accept.")

            st.subheader("Step 4 — NAICS Secondary Check (informational, does not override)")
            naics_sims = q_emb @ naics_embeddings.T
            best_naics_idx = int(np.argmax(naics_sims[0]))
            best_naics_score = naics_sims[0][best_naics_idx]
            naics_row = naics_df.iloc[best_naics_idx]
            naics_cob = naics_row["Hiscox_COB"]
            naics_desc = naics_row["NAICS_Description"]

            st.write(f"Closest NAICS description found: *\"{naics_desc}\"* (similarity: {best_naics_score:.3f})")
            st.write(f"That NAICS row's known outcome: **{naics_cob}**")

            if naics_cob == top_cob:
                st.success("Agreement: NAICS check confirms the same COB as semantic matching. Confidence boost — this strengthens the case for auto-accept.")
            elif naics_cob == "OOA" and top_cob != "OOA":
                if best_naics_score >= 0.85:
                    st.error("Strong disagreement: NAICS match is near-exact and says OOA, but semantic matching found an in-appetite COB. The NAICS answer should likely be trusted here — flag for review leaning OOA.")
                else:
                    st.warning("Disagreement: semantic matching found an in-appetite COB, but the closest NAICS description is OOA. Worth flagging for review.")
            else:
                if best_naics_score >= 0.85:
                    st.error(f"Strong disagreement: NAICS match is near-exact (similarity {best_naics_score:.2f}) and points to \"{naics_cob}\" — this is a curated, high-confidence answer and should likely be trusted over semantic matching's \"{top_cob}\" (similarity {top_cob_score:.2f}).")
                elif top_cob_score >= 0.70 and best_naics_score < 0.70:
                    st.info(f"Mild disagreement, but semantic matching scored well ({top_cob_score:.2f}) while the NAICS match was weaker ({best_naics_score:.2f}) — semantic match \"{top_cob}\" is probably more trustworthy here, unless a known business rule (e.g. product availability) favors the NAICS answer.")
                else:
                    st.warning(f"Genuine disagreement: semantic matching says \"{top_cob}\" ({top_cob_score:.2f}), NAICS check points to \"{naics_cob}\" ({best_naics_score:.2f}). Neither is clearly stronger — flag for review, show both candidates.")
