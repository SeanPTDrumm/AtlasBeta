import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Hiscox Digital Atlas", layout="wide")


# ---------------------------------------------------------------------------
# DATA LOADING  (unchanged)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_data():
    cobs = pd.read_csv("atlas_cobs.csv")
    naics = pd.read_csv("atlas_naics.csv")
    rules = pd.read_csv("atlas_rules.csv")
    return cobs, naics, rules


@st.cache_resource
def load_verified():
    """Load the human-verified answer bank: exact/near-exact phrases with
    confirmed correct outcomes, built from real graded batches."""
    verified = {}
    try:
        bank = pd.read_csv("atlas_verified_bank.csv")
        for _, r in bank.iterrows():
            verified[str(r["Partner_Name"]).strip().lower()] = (r["Verified_COB"], "verified_bank")
    except FileNotFoundError:
        pass
    try:
        outliers = pd.read_csv("atlas_verified_outliers.csv")
        for _, r in outliers.iterrows():
            verified[str(r["Partner_Name_Pattern"]).strip().lower()] = (r["Verified_COB"], "verified_outlier")
    except FileNotFoundError:
        pass
    return verified


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
verified_bank = load_verified()
vec, appetite_clf = train_appetite_classifier(naics)

with st.spinner("Loading models (first run only, may take a minute)..."):
    model = load_semantic_model()
    cob_texts, cob_embeddings = encode_cobs(model, cobs)
    naics_df, naics_embeddings = encode_naics(model, naics)

cob_names = cobs["Hiscox_COB"].tolist()

# Fast lookup of COB attributes (GL/PL/BOP/Cyber, industry code, definition)
# keyed by lowercased COB name, built once from atlas_cobs.csv.
COB_ATTR = {}
for _, _r in cobs.iterrows():
    COB_ATTR[str(_r["Hiscox_COB"]).strip().lower()] = {
        "COB": _r["Hiscox_COB"],
        "Group": _r.get("COB_Group", ""),
        "GL": str(_r.get("GL", "")).strip(),
        "PL": str(_r.get("PL", "")).strip(),
        "BOP": str(_r.get("BOP", "")).strip(),
        "Cyber": str(_r.get("Cyber", "")).strip(),
        "Industry_Code": _r.get("Full_Industry_Code", ""),
        "Definition": _r.get("Definition", ""),
    }

STEP0_STOP_THRESHOLD = 0.80
OVERRIDE_NAME_MATCH_THRESHOLD = 0.75
SEMANTIC_OWN_CONFIDENCE_THRESHOLD = 0.60
SEMANTIC_AUTO_TRUST_THRESHOLD = 0.70
WEAK_WINNER_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.45

# COBs known to be structurally ambiguous - a high semantic score alone isn't
# enough to skip review, because the SAME wording can mean genuinely different
# things (e.g. "Project Manager" alone is usually business/general project
# management, but in construction/architecture context should be "Agency
# Construction Manager" instead - the two are easy to confuse and always
# deserve a second look regardless of match confidence).
ALWAYS_FLAG_COBS = ["Project management", "Agency construction manager"]


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


# Hardcoded, CSV-independent safety net for guide-sourced rules. These do not
# depend on atlas_rules.csv loading correctly - they are checked directly
# in code so they cannot fail due to any file-loading issue.
HARDCODED_OOA_PHRASES = [
    "design build", "design-build", "design builder",
    "home inspector", "home inspection",
    "independent movie producer", "independent film producer",
    "documentary producer", "television producer",
    "skyscraper photography", "underwater photography",
]
HARDCODED_INAPPETITE_PHRASES = [
    "building inspector", "building code inspection", "building inspection",
]


def check_hardcoded_rules(name):
    nl = name.lower()
    for phrase in HARDCODED_INAPPETITE_PHRASES:
        if phrase in nl:
            return "In-Appetite", phrase
    for phrase in HARDCODED_OOA_PHRASES:
        if phrase in nl:
            return "OOA", phrase
    return None, None


MOJIBAKE_FIXES = {
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\x9d": '"',
    "\u00e2\u20ac\x93": "-",
    "\u00e2\u20ac\x94": "-",
    "\u00e2\u20ac": "-",
}


def clean_text(name):
    """Repair common UTF-8/Windows-1252 encoding corruption (mojibake) seen
    in real partner data, e.g. Manufacturer's -> Manufacturer's."""
    cleaned = name
    for bad, good in MOJIBAKE_FIXES.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


# ---------------------------------------------------------------------------
# PIPELINE ENGINE  (logic unchanged — only the UI around it is new)
# ---------------------------------------------------------------------------
def run_pipeline(query):
    query = clean_text(query)
    row = {"Partner_Name": query}
    # Step -2: hardcoded, CSV-independent guide-sourced rules (checked first,
    # cannot fail due to any file-loading issue)
    hardcoded_direction, hardcoded_phrase = check_hardcoded_rules(query)
    if hardcoded_direction == "OOA":
        row["Final_Stage"] = "Hardcoded_Rule_OOA_Stop"
        row["Recommended_COB"] = "OOA"
        row["Needs_Review"] = False
        row["Review_Confidence"] = f'High - hardcoded guide rule matched: "{hardcoded_phrase}"'
        row["Step0_Prediction"] = ""
        row["Step0_Confidence"] = ""
        row["Step0_Short_Input"] = ""
        row["Step0_Override"] = ""
        row["Vague_Input_Flag"] = ""
        row["Rule_Hit"] = hardcoded_phrase
        row["Top1_COB"] = ""
        row["Top1_Score"] = ""
        row["Top2_COB"] = ""
        row["Top3_COB"] = ""
        row["NAICS_Closest_Desc"] = ""
        row["NAICS_COB"] = ""
        row["NAICS_Score"] = ""
        row["Disagreement_Category"] = ""
        return row
    # Step -1: has a human already verified the exact answer for this phrase?
    lookup_key = query.strip().lower()
    if lookup_key in verified_bank:
        verified_cob, source = verified_bank[lookup_key]
        row["Final_Stage"] = "Verified_Bank_Match"
        row["Recommended_COB"] = verified_cob
        row["Needs_Review"] = False
        row["Review_Confidence"] = f"Very High - exact match to human-verified example ({source})"
        row["Step0_Prediction"] = ""
        row["Step0_Confidence"] = ""
        row["Step0_Short_Input"] = ""
        row["Step0_Override"] = ""
        row["Vague_Input_Flag"] = ""
        row["Rule_Hit"] = ""
        row["Top1_COB"] = ""
        row["Top1_Score"] = ""
        row["Top2_COB"] = ""
        row["Top3_COB"] = ""
        row["NAICS_Closest_Desc"] = ""
        row["NAICS_COB"] = ""
        row["NAICS_Score"] = ""
        row["Disagreement_Category"] = ""
        return row
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
        carve_in_check = check_rules(query, rules)
        has_carve_in = carve_in_check is not None and carve_in_check["Rule_Type"] == "sector_carve_in"
        if best_score < OVERRIDE_NAME_MATCH_THRESHOLD and not has_carve_in:
            row["Final_Stage"] = "Step0_OOA_Stop"
            row["Recommended_COB"] = "OOA"
            empty_match_fields()
            return row
        elif has_carve_in:
            row["Step0_Override"] = f"Overridden - carve-in rule matched: \"{carve_in_check['Pattern_or_Phrase']}\""
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
        if row["Disagreement_Category"] == "Trust_Semantic" and top_score >= SEMANTIC_AUTO_TRUST_THRESHOLD and top_cob not in ALWAYS_FLAG_COBS:
            # Evidence: when semantic's own score is high (>=0.72), it has been
            # reliably correct (0/11 wrong in a real fresh-batch check) - skip
            # the flag, same principle as the OOA-vs-NAICS rule above.
            row["Needs_Review"] = False
            row["Review_Confidence"] = f"High - semantic match strongly confident on its own (score {winning_score:.2f})"
        elif winning_score >= WEAK_WINNER_THRESHOLD:
            if row["Disagreement_Category"] == "Trust_NAICS" and naics_cob == "OOA":
                # Evidence: OOA-vs-in-appetite disagreements are usually clean,
                # obvious calls (80% felt unnecessary to flag in real grading).
                # COB-vs-COB disagreements are genuinely trickier - keep flagging those.
                row["Needs_Review"] = False
                row["Review_Confidence"] = f"High - NAICS confidently says OOA (winning score {winning_score:.2f})"
            else:
                row["Review_Confidence"] = f"Quick confirm - recommendation likely correct (~87% historically), winning score {winning_score:.2f}"
        else:
            row["Review_Confidence"] = f"Genuinely unclear - even the winning answer scored weak ({winning_score:.2f}) - full manual review needed"
    return row


# ---------------------------------------------------------------------------
# PRESENTATION HELPERS  (new — translate engine output into Source-of-Truth UI)
# ---------------------------------------------------------------------------
def user_status(result):
    """Translate the internal confidence/review logic into a business-facing
    status. Returns (emoji, label, help_text)."""
    conf = str(result.get("Review_Confidence", "") or "")
    needs = bool(result.get("Needs_Review", False))
    if "Genuinely unclear" in conf:
        return ("\U0001F534", "Needs Review",
                "Atlas could not reach a reliable classification. Send to Underwriting Management.")
    if needs:  # quick-confirm bucket
        return ("\U0001F7E1", "Please Confirm",
                "Atlas has a recommendation and it's probably right. Take a quick look before using it.")
    return ("\U0001F7E2", "Recommended",
            "Atlas is confident in this match. Ready to use.")


def cob_attributes(cob_name):
    """Return the LOB eligibility, industry code and definition for a COB name.
    Handles OOA and any name not present in the COB table gracefully."""
    if not cob_name or str(cob_name).strip().upper() == "OOA":
        return {
            "COB": "Out of Appetite (OOA)",
            "Group": "",
            "GL": "No", "PL": "No", "BOP": "No", "Cyber": "No",
            "Industry_Code": "\u2014",
            "Definition": "This business falls outside Hiscox digital appetite.",
            "is_ooa": True,
        }
    attr = COB_ATTR.get(str(cob_name).strip().lower())
    if attr is None:
        return {
            "COB": cob_name, "Group": "",
            "GL": "\u2014", "PL": "\u2014", "BOP": "\u2014", "Cyber": "\u2014",
            "Industry_Code": "\u2014",
            "Definition": "", "is_ooa": False,
        }
    out = dict(attr)
    out["is_ooa"] = False
    return out


def yn_icon(v):
    return "\u2705" if str(v).strip().lower() in ("yes", "y", "true") else "\u274C"


def lob_table_md(attr):
    return (
        "| Product | Eligible |\n|---|---|\n"
        f"| GL | {yn_icon(attr['GL'])} |\n"
        f"| PL | {yn_icon(attr['PL'])} |\n"
        f"| BOP | {yn_icon(attr['BOP'])} |\n"
        f"| Cyber | {yn_icon(attr['Cyber'])} |"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; letter-spacing:0.5px;'>Hiscox Digital Atlas</h1>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='border:none; border-top:2px solid #d0021b; margin-top:0;'>",
            unsafe_allow_html=True)

mode = st.radio("Mode", ["Search", "Upload a Batch File"], horizontal=True,
                label_visibility="collapsed")
st.write("")

# ------------------------------- SEARCH -----------------------------------
if mode == "Search":
    query = st.text_input(
        "Enter a partner term or business description",
        placeholder="e.g., Business Consulting, Engineering, Landscaping",
    )
    if query:
        result = run_pipeline(query)
        emoji, label, help_text = user_status(result)
        attr = cob_attributes(result["Recommended_COB"])

        # ---- Source of Truth result card ----
        st.markdown(f"## {attr['COB']}")
        st.markdown(f"### {emoji} {label}")
        st.caption(help_text)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Industry Code**")
            st.markdown(f"`{attr['Industry_Code']}`")
            if attr["Group"]:
                st.markdown("**COB Group**")
                st.markdown(attr["Group"])
        with c2:
            st.markdown("**Eligible Products**")
            st.markdown(lob_table_md(attr))

        if attr["Definition"]:
            st.markdown("**Definition**")
            st.write(attr["Definition"])

        # ---- Details (SME mode) tucked away ----
        with st.expander("\u25BC Details (Underwriting / SME view)"):
            st.write(f"**Final stage:** {result.get('Final_Stage', '')}")
            st.write(f"**Internal review note:** {result.get('Review_Confidence', '')}")
            if result.get("Step0_Prediction"):
                st.write(
                    f"**Appetite classifier:** {result['Step0_Prediction']} "
                    f"(confidence {result.get('Step0_Confidence', '')})"
                )
            if result.get("Step0_Override"):
                st.write(f"**Step 0 override:** {result['Step0_Override']}")
            if result.get("Rule_Hit"):
                st.write(f"**Rule hit:** {result['Rule_Hit']}")
            if result.get("Top1_COB"):
                st.write("**Top semantic candidates**")
                st.write(f"1. {result['Top1_COB']}  (score {result.get('Top1_Score', '')})")
                if result.get("Top2_COB"):
                    st.write(f"2. {result['Top2_COB']}")
                if result.get("Top3_COB"):
                    st.write(f"3. {result['Top3_COB']}")
            if result.get("NAICS_COB"):
                st.write(
                    f"**NAICS cross-check:** {result['NAICS_COB']} "
                    f"(closest: \"{result.get('NAICS_Closest_Desc', '')}\", "
                    f"score {result.get('NAICS_Score', '')})"
                )

# ------------------------------- BATCH ------------------------------------
else:
    st.subheader("Upload a Batch File")
    uploaded_file = st.file_uploader(
        "Drag and drop a CSV or Excel file with partner terms (one column)",
        type=["csv", "xlsx"],
    )
    names = []
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            upload_df = pd.read_csv(uploaded_file)
        else:
            upload_df = pd.read_excel(uploaded_file)
        st.write("Preview:")
        st.dataframe(upload_df.head())
        col_choice = st.selectbox("Which column has the partner terms?", upload_df.columns.tolist())
        names = upload_df[col_choice].dropna().astype(str).tolist()
        st.write(f"Found {len(names)} terms.")

    if st.button("Run batch") and names:
        if len(names) > 300:
            st.warning(f"You have {len(names)} terms - only processing the first 300 to keep this responsive.")
            names = names[:300]
        progress = st.progress(0)
        clean_rows = []
        tech_rows = []
        for i, name in enumerate(names):
            res = run_pipeline(name)
            emoji, label, _ = user_status(res)
            attr = cob_attributes(res["Recommended_COB"])
            clean_rows.append({
                "Partner Description": res["Partner_Name"],
                "Recommended COB": attr["COB"],
                "Industry Code": attr["Industry_Code"],
                "GL": "Y" if str(attr["GL"]).strip().lower() in ("yes", "y", "true") else "N",
                "PL": "Y" if str(attr["PL"]).strip().lower() in ("yes", "y", "true") else "N",
                "BOP": "Y" if str(attr["BOP"]).strip().lower() in ("yes", "y", "true") else "N",
                "Cyber": "Y" if str(attr["Cyber"]).strip().lower() in ("yes", "y", "true") else "N",
                "Status": f"{emoji} {label}",
            })
            tech_rows.append(res)
            progress.progress((i + 1) / len(names))
        st.session_state["batch_clean"] = pd.DataFrame(clean_rows)
        st.session_state["batch_tech"] = pd.DataFrame(tech_rows)

    if "batch_clean" in st.session_state:
        clean_df = st.session_state["batch_clean"]

        # Quick triage summary
        n_total = len(clean_df)
        n_review = clean_df["Status"].str.contains("Needs Review").sum()
        n_confirm = clean_df["Status"].str.contains("Please Confirm").sum()
        n_reco = clean_df["Status"].str.contains("Recommended").sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", n_total)
        m2.metric("\U0001F7E2 Recommended", int(n_reco))
        m3.metric("\U0001F7E1 Please Confirm", int(n_confirm))
        m4.metric("\U0001F534 Needs Review", int(n_review))

        st.subheader("Results")
        st.dataframe(clean_df, use_container_width=True, hide_index=True)

        # Clean, user-facing download
        clean_csv = clean_df.to_csv(index=False)
        st.download_button(
            "Download results (CSV)", clean_csv, "atlas_results.csv", "text/csv"
        )

        # Optional technical export for SMEs
        with st.expander("Download technical results (Underwriting / SME view)"):
            tech_csv = st.session_state["batch_tech"].to_csv(index=False)
            st.download_button(
                "Download technical CSV", tech_csv,
                "atlas_results_technical.csv", "text/csv"
            )
