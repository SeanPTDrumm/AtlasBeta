import streamlit as st
import pandas as pd
import re
import html
import base64
import math
from pathlib import Path
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
# ATLAS PRESENTATION + FINAL UI
# Visual source of truth:
#   Atlas_Search_Home_FINAL.png
#   Atlas_Single_Search_Result_FINAL.png
#   Atlas_Batch_Upload_Empty_FINAL.png
#   Atlas_Batch_Review_FINAL.png
# ---------------------------------------------------------------------------

ATLAS_RED = "#ef1b24"
ATLAS_GREEN = "#75c934"
ATLAS_AMBER = "#f0aa16"
ATLAS_BG = "#020101"
ATLAS_PANEL = "#090909"
ATLAS_LINE = "#2b2825"
ATLAS_TEXT = "#f5f5f5"
ATLAS_MUTED = "#9a9a9a"


def _b64(path):
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("ascii")


_LOGO_B64 = _b64("Atlas_Logo_A.png")

st.markdown(f"""
<style>
:root{{--red:{ATLAS_RED};--green:{ATLAS_GREEN};--amber:{ATLAS_AMBER};--bg:{ATLAS_BG};--panel:{ATLAS_PANEL};--line:{ATLAS_LINE};--text:{ATLAS_TEXT};--muted:{ATLAS_MUTED};}}
html,body,[class*="css"]{{font-family:Arial,Helvetica,sans-serif}}
.stApp{{background:var(--bg);color:var(--text)}}
[data-testid="stHeader"]{{background:transparent;height:0}}
[data-testid="stToolbar"],#MainMenu,footer{{visibility:hidden}}
.block-container{{max-width:1360px;padding:1.15rem 2.2rem 1rem}}
p,h1,h2,h3,label,span{{color:var(--text)}}
/* navigation */
div[role="radiogroup"]{{display:flex;justify-content:center;gap:2.4rem;margin:.1rem auto 1rem}}
div[role="radiogroup"] label{{padding:.65rem .35rem .8rem;border-bottom:3px solid transparent;font-weight:750;letter-spacing:.03em}}
div[role="radiogroup"] label:has(input:checked){{border-bottom-color:var(--red)}}
div[role="radiogroup"] [data-testid="stMarkdownContainer"] p{{font-size:1rem}}
/* regular controls */
[data-testid="stTextInput"] input,[data-testid="stSelectbox"]>div>div,[data-baseweb="select"]>div{{background:#070707!important;color:#fff!important;border-color:#3a3733!important}}
[data-testid="stTextInput"] input{{height:50px;border-radius:9px;font-size:.96rem}}
[data-testid="stTextInput"] input:focus{{border-color:var(--red)!important;box-shadow:0 0 0 1px var(--red)!important}}
.stButton>button,.stDownloadButton>button{{background:var(--red);color:#fff;border:1px solid var(--red);border-radius:7px;font-weight:800;min-height:40px}}
.stButton>button:hover,.stDownloadButton>button:hover{{background:#ff3138;color:#fff;border-color:#ff3138}}
/* header + footer */
.atlas-header{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.1rem}}
.atlas-logo-small{{width:148px;height:auto;display:block}}
.atlas-help{{color:#fff;font-size:.95rem;padding-top:.35rem}} .atlas-help span{{display:inline-flex;border:1px solid #fff;border-radius:50%;width:22px;height:22px;align-items:center;justify-content:center;margin-right:.45rem}}
.atlas-footer{{border-top:2px solid var(--red);padding:1.1rem .1rem .25rem;margin-top:1.35rem;display:flex;align-items:center;justify-content:space-between;color:#aaa;font-size:.86rem}}
.atlas-footer-brand{{color:#eee;font-weight:800;font-size:1.05rem}} .atlas-footer-brand b{{color:var(--red);margin-right:.35rem}}
/* search home */
.home-wrap{{min-height:560px;display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:-3.2rem}}
.home-logo{{width:225px;max-height:205px;object-fit:contain;margin-bottom:2.25rem}}
.home-search{{max-width:900px;margin:0 auto}}
/* cards */
.atlas-card{{max-width:1160px;margin:.15rem auto 0;border:1px solid #39352f;border-radius:13px;background:linear-gradient(145deg,#090909,#050505);padding:1.7rem 1.9rem 1.05rem;box-shadow:0 12px 32px rgba(0,0,0,.32)}}
.result-top{{display:flex;justify-content:space-between;gap:2rem;align-items:flex-start}}
.result-cob{{font-size:1.9rem;font-weight:800;letter-spacing:-.025em;line-height:1.1;margin:.1rem 0 .45rem}}
.result-group{{font-size:.98rem;color:#8b8b8b}}
.kicker{{font-size:.72rem;text-transform:uppercase;color:#858585;font-weight:800;letter-spacing:.08em}}
.status-box{{border:1px solid currentColor;border-radius:8px;padding:.62rem .85rem;font-size:.95rem;font-weight:850;min-width:190px;text-align:center}}
.status-clear{{color:var(--green)}} .status-likely{{color:var(--amber)}} .status-unclear{{color:var(--red)}}
.elig-title{{margin:1.8rem 0 .8rem;color:#979797;font-weight:850;font-size:.8rem}}
.elig-grid{{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid #39352f;padding-bottom:1.05rem}}
.elig{{text-align:center;border-right:1px solid #302d29}} .elig:last-child{{border-right:none}}
.elig-name{{font-weight:850;font-size:.96rem}} .elig-icon{{font-size:1.45rem;margin:.4rem 0 .15rem}} .elig-value{{font-size:.96rem;font-weight:850}}
.yes{{color:var(--green)}} .no{{color:#d12b31}}
.definition{{padding:1rem 0 .5rem;border-bottom:1px solid #39352f;color:#b3b3b3;line-height:1.4;font-size:.92rem}}
.details-row{{text-align:right;color:#858585;padding:.65rem 0 0;font-size:.86rem}}
/* uploader */
.upload-shell{{max-width:920px;margin:2.1rem auto 3.2rem;border:1px solid #39352f;border-radius:14px;background:#070707;padding:1.45rem}}
[data-testid="stFileUploader"]{{background:#050505;border:1px dashed #8e8a84;border-radius:12px;padding:1.75rem .9rem}}
[data-testid="stFileUploaderDropzone"]{{background:transparent;border:0;min-height:205px}}
[data-testid="stFileUploaderDropzoneInstructions"] span{{color:#f4f4f4!important;font-size:1.05rem;font-weight:750}}
[data-testid="stFileUploaderDropzoneInstructions"] small{{color:#999!important}}
[data-testid="stFileUploader"] button{{background:#090909!important;color:#fff!important;border:1px solid var(--red)!important}}
.file-ready{{border:1px solid #39352f;border-radius:12px;background:#070707;padding:1.2rem 1.35rem;margin:.5rem 0 1rem}}
/* batch */
.batch-frame{{border:1px solid #39352f;border-radius:14px;background:#070707;padding:1rem 1rem .8rem}}
.batch-file{{display:flex;align-items:center;gap:1rem;padding:.1rem 0 1rem;border-bottom:1px solid #2c2925}}
.batch-filename{{font-size:1rem;font-weight:750}} .batch-count{{color:#aaa}}
.metric-card{{border:1px solid #39352f;border-radius:10px;background:#090909;text-align:center;padding:.8rem .45rem;min-height:106px}}
.metric-n{{font-size:1.75rem;font-weight:850}} .metric-label{{font-size:.82rem;font-weight:800;margin:.28rem 0}} .metric-sub{{font-size:.74rem;color:#999}}
.instruction{{border:1px solid #39352f;border-radius:9px;padding:.8rem 1rem;color:#c4c4c4;margin:1rem 0 .75rem;font-size:.9rem}}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{{border:1px solid #39352f;border-radius:9px;overflow:hidden}}
[data-testid="stMetric"]{{background:#090909;border:1px solid #39352f;border-radius:10px;padding:.8rem}}
[data-testid="stExpander"]{{background:#080808;border:1px solid #39352f;border-radius:9px}}
/* compact utility row */
.utility-row{{margin-top:.35rem}}
@media(max-width:800px){{.block-container{{padding:1rem}}.result-top{{display:block}}.status-box{{margin-top:1rem}}.elig-grid{{grid-template-columns:repeat(2,1fr);gap:1rem}}.elig{{border:0}}.home-logo{{width:200px}}.atlas-logo-small{{width:130px}}}}
</style>
""",unsafe_allow_html=True)


def _safe(v):
    return html.escape(str(v if v is not None else ""))


def atlas_status(result):
    note = str(result.get("Review_Confidence", "") or "")
    if "Genuinely unclear" in note:
        return "Unclear", "unclear"
    if bool(result.get("Needs_Review", False)):
        return "Likely Match", "likely"
    return "Clear Match", "clear"


def cob_attributes(cob_name):
    if not cob_name or str(cob_name).strip().upper() == "OOA":
        return {"COB":"OOA","Group":"","GL":"No","PL":"No","BOP":"No","Cyber":"No","Industry_Code":"","Definition":"Outside Hiscox digital appetite.","is_ooa":True}
    found = COB_ATTR.get(str(cob_name).strip().lower())
    if found is None:
        return {"COB":str(cob_name),"Group":"","GL":"—","PL":"—","BOP":"—","Cyber":"—","Industry_Code":"","Definition":"","is_ooa":False}
    out = dict(found)
    out["is_ooa"] = False
    return out


def _eligible(v):
    return str(v).strip().lower() in ("yes","y","true")


def header():
    logo = f'data:image/png;base64,{_LOGO_B64}' if _LOGO_B64 else ''
    logo_html = f'<img src="{logo}" class="atlas-logo-small">' if logo else '<div class="atlas-footer-brand">HISCOX<br>ATLAS</div>'
    st.markdown(f'<div class="atlas-header">{logo_html}<div class="atlas-help"><span>?</span>Help</div></div>',unsafe_allow_html=True)


def footer():
    st.markdown('<div class="atlas-footer"><div class="atlas-footer-brand"><b>♦</b>HISCOX</div><div>Internal Use Only&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;v1.0</div></div>',unsafe_allow_html=True)


def navigation(default="Search", key="main_nav"):
    return st.radio("Navigation", ["Search","Batch Upload"], index=0 if default=="Search" else 1, horizontal=True, label_visibility="collapsed", key=key)


def status_html(label, cls):
    icon = "✓" if cls == "clear" else ("~" if cls == "likely" else "!")
    return f'<div class="kicker">Atlas Status</div><div class="status-box status-{cls}">{icon}&nbsp;&nbsp;{_safe(label).upper()}</div>'


def eligibility_html(attr):
    parts=[]
    for name in ["GL","PL","BOP","Cyber"]:
        ok=_eligible(attr[name])
        parts.append(f'<div class="elig"><div class="elig-name {"yes" if ok else "no"}">{name.upper()}</div><div class="elig-icon {"yes" if ok else "no"}">{"✓" if ok else "×"}</div><div class="elig-value {"yes" if ok else "no"}">{"Yes" if ok else "No"}</div></div>')
    return ''.join(parts)


def render_search_result(query):
    result = run_pipeline(query)
    attr = cob_attributes(result["Recommended_COB"])
    label, cls = atlas_status(result)
    group = f'<div class="result-group">{_safe(attr.get("Group",""))}</div>' if attr.get("Group") else ''
    st.markdown(f"""<div class="atlas-card"><div class="result-top"><div><div class="result-cob">{_safe(attr["COB"])}</div>{group}</div><div>{status_html(label,cls)}</div></div><div class="elig-title">DIGITAL PRODUCT ELIGIBILITY</div><div class="elig-grid">{eligibility_html(attr)}</div><div class="definition"><div class="kicker" style="margin-bottom:.65rem">Description</div>{_safe(attr.get("Definition",""))}</div><div class="details-row">›&nbsp;&nbsp;View classification details</div></div>""",unsafe_allow_html=True)
    with st.expander("Classification details"):
        if attr.get("Industry_Code"):
            st.write(f"**Industry code:** {attr['Industry_Code']}")
        st.write(f"**Atlas review logic:** {result.get('Review_Confidence','')}")
        if result.get("Top1_COB"):
            st.write(f"**Top match:** {result.get('Top1_COB')} ({result.get('Top1_Score','')})")
        if result.get("Top2_COB"):
            st.write(f"**Alternative:** {result.get('Top2_COB')}")
        if result.get("NAICS_COB"):
            st.write(f"**NAICS cross-check:** {result.get('NAICS_COB')} ({result.get('NAICS_Score','')})")


def make_batch_row(result, row_id):
    attr = cob_attributes(result["Recommended_COB"])
    label, _ = atlas_status(result)
    action = "Keep Match" if label != "Unclear" else "Send to UWM"
    return {
        "_RowID": row_id,
        "Partner Term": result["Partner_Name"],
        "Hiscox COB": attr["COB"],
        "Industry Code": attr.get("Industry_Code", "") if not attr.get("is_ooa") else "",
        "GL": "✓" if _eligible(attr["GL"]) else "×",
        "PL": "✓" if _eligible(attr["PL"]) else "×",
        "BOP": "✓" if _eligible(attr["BOP"]) else "×",
        "Cyber": "✓" if _eligible(attr["Cyber"]) else "×",
        "Atlas Status": label,
        "Action": action,
        "Final COB": attr["COB"],
    }


def result_export(df):
    output = df.copy()
    for idx, row in output.iterrows():
        final_cob = row["Final COB"] if row["Action"] == "Change COB" else row["Hiscox COB"]
        if row["Action"] == "Send to UWM":
            final_cob = row["Final COB"]
        final_attr = cob_attributes(final_cob)
        output.at[idx,"Final COB"] = final_cob
        output.at[idx,"Industry Code"] = "" if final_attr.get("is_ooa") else final_attr.get("Industry_Code","")
        output.at[idx,"GL"] = "Y" if _eligible(final_attr["GL"]) else "N"
        output.at[idx,"PL"] = "Y" if _eligible(final_attr["PL"]) else "N"
        output.at[idx,"BOP"] = "Y" if _eligible(final_attr["BOP"]) else "N"
        output.at[idx,"Cyber"] = "Y" if _eligible(final_attr["Cyber"]) else "N"
    return output.drop(columns=["_RowID"],errors="ignore")


# Session defaults
if "atlas_mode" not in st.session_state:
    st.session_state.atlas_mode = "Search"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# SEARCH HOME differs intentionally from working screens
if st.session_state.atlas_mode == "Search" and not st.session_state.search_query:
    st.markdown('<div class="atlas-help" style="text-align:right"><span>?</span>Help</div>',unsafe_allow_html=True)
    logo = f'data:image/png;base64,{_LOGO_B64}' if _LOGO_B64 else ''
    logo_html = f'<img src="{logo}" class="home-logo">' if logo else '<div class="result-cob">HISCOX ATLAS</div>'
    st.markdown(f'<div class="home-wrap">{logo_html}</div>',unsafe_allow_html=True)
    # Pull the search form upward into the visual center.
    st.markdown('<style>.home-wrap{min-height:295px;margin-top:0;margin-bottom:-1.75rem}</style>',unsafe_allow_html=True)
    c1,c2 = st.columns([8,1])
    with c1:
        q = st.text_input("Search",placeholder="Search a partner term or business description",label_visibility="collapsed",key="home_q")
    with c2:
        go = st.button("→",use_container_width=True,key="home_go")
    nav = navigation("Search","home_nav")
    if nav == "Batch Upload":
        st.session_state.atlas_mode = "Batch Upload"
        st.rerun()
    if (go or q) and q.strip():
        st.session_state.search_query = q.strip()
        st.rerun()
    footer()

else:
    header()
    nav = navigation(st.session_state.atlas_mode,"work_nav")
    if nav != st.session_state.atlas_mode:
        st.session_state.atlas_mode = nav
        if nav == "Batch Upload":
            st.session_state.search_query = ""
        st.rerun()

    if st.session_state.atlas_mode == "Search":
        c1,c2=st.columns([8,1])
        with c1:
            q=st.text_input("Search",value=st.session_state.search_query,placeholder="Search a partner term or business description",label_visibility="collapsed",key="result_q")
        with c2:
            go=st.button("→",use_container_width=True,key="result_go")
        if go and q.strip():
            st.session_state.search_query=q.strip()
        if q.strip():
            render_search_result(q.strip())
        footer()

    else:
        # Empty/loded states use the native file uploader, styled to visually match the approved prototype.
        uploaded = None
        if "batch_results" not in st.session_state:
            st.markdown('<div class="upload-shell">',unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload a partner list",type=["csv"],help="Drag and drop a CSV file or browse files.",key="batch_file")
            st.markdown('</div>',unsafe_allow_html=True)
        if uploaded is not None and "batch_results" not in st.session_state:
            raw=pd.read_csv(uploaded)
            st.markdown(f'<div class="file-ready"><b>{_safe(uploaded.name)}</b>&nbsp;&nbsp;&nbsp;<span class="batch-count">{len(raw):,} terms</span></div>',unsafe_allow_html=True)
            left,right=st.columns([3,1])
            with left:
                selected_col=st.selectbox("Partner term column",raw.columns.tolist(),key="batch_column")
            with right:
                st.write("")
                run=st.button("RUN ATLAS",use_container_width=True,key="run_batch")
            if run:
                names=raw[selected_col].dropna().astype(str).tolist()
                if len(names)>300:
                    st.warning(f"Only the first 300 of {len(names)} terms will be processed in this session.")
                    names=names[:300]
                bar=st.progress(0,text="Classifying partner terms...")
                rows=[]; technical=[]
                for i,name in enumerate(names):
                    res=run_pipeline(name); rows.append(make_batch_row(res,i)); technical.append(res)
                    bar.progress((i+1)/len(names),text=f"Classifying {i+1} of {len(names)}")
                bar.empty()
                st.session_state.batch_results=pd.DataFrame(rows)
                st.session_state.batch_technical=pd.DataFrame(technical)
                st.session_state.batch_filename=uploaded.name
                st.rerun()

        if "batch_results" in st.session_state:
            df=st.session_state.batch_results.copy()
            filename=st.session_state.get("batch_filename","partner_terms.csv")
            top1,top2=st.columns([5,1])
            with top1:
                st.markdown(f'<div class="batch-frame"><div class="batch-file"><div class="batch-filename">▣&nbsp;&nbsp;{_safe(filename)}</div><div class="batch-count">{len(df):,} terms</div></div>',unsafe_allow_html=True)
            with top2:
                if st.button("Change file",key="change_batch_file"):
                    for k in ["batch_results","batch_technical","batch_filename","batch_file"]:
                        st.session_state.pop(k,None)
                    st.rerun()
            c1,c2,c3,c4=st.columns(4)
            counts={s:int((df["Atlas Status"]==s).sum()) for s in ["Clear Match","Likely Match","Unclear"]}
            cards=[(c1,counts["Clear Match"],"CLEAR MATCH","No action required","clear"),(c2,counts["Likely Match"],"LIKELY MATCH","Review recommended","likely"),(c3,counts["Unclear"],"UNCLEAR","UWM review needed","unclear"),(c4,len(df),"TOTAL RESULTS","All classifications","total")]
            for col,n,label,sub,cls in cards:
                color=ATLAS_GREEN if cls=="clear" else (ATLAS_AMBER if cls=="likely" else (ATLAS_RED if cls=="unclear" else "#f5f5f5"))
                with col: st.markdown(f'<div class="metric-card"><div class="metric-n" style="color:{color}">{n}</div><div class="metric-label">{label}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="instruction">ⓘ&nbsp;&nbsp;Check the Likely Match and Unclear results. Keep the match, select another Hiscox COB, or send it to UWM.</div>',unsafe_allow_html=True)

            f1,f2,f3=st.columns([2.2,1.2,.8])
            with f1:
                status_filter=st.radio("Status",["All Results","Clear Match","Likely Match","Unclear"],horizontal=True,label_visibility="collapsed",key="status_filter")
            with f2:
                term_filter=st.text_input("Filter",placeholder="Filter batch results",label_visibility="collapsed",key="term_filter")
            with f3:
                show_code=st.checkbox("Industry Code",value=False,key="show_code")

            view=df.copy()
            if status_filter!="All Results": view=view[view["Atlas Status"]==status_filter]
            if term_filter.strip():
                needle=term_filter.strip().lower()
                mask=view["Partner Term"].str.lower().str.contains(needle,na=False)|view["Hiscox COB"].str.lower().str.contains(needle,na=False)|view["Final COB"].astype(str).str.lower().str.contains(needle,na=False)
                view=view[mask]

            # Ten rows by default, with direct Previous / Next navigation.
            page_size=st.selectbox("Rows per page",[10,25,50],index=0,key="page_size")
            pages=max(1,math.ceil(len(view)/page_size))

            # Reset to page 1 when the result set or page size changes.
            pagination_signature=(status_filter,term_filter.strip().lower(),page_size,len(view))
            if st.session_state.get("pagination_signature") != pagination_signature:
                st.session_state.batch_page=1
                st.session_state.pagination_signature=pagination_signature
            if "batch_page" not in st.session_state:
                st.session_state.batch_page=1
            st.session_state.batch_page=max(1,min(st.session_state.batch_page,pages))

            previous_disabled=st.session_state.batch_page <= 1
            next_disabled=st.session_state.batch_page >= pages
            previous_col,page_col,next_col=st.columns([1,2,1])
            with previous_col:
                if st.button("← Previous",disabled=previous_disabled,use_container_width=True,key="previous_page"):
                    st.session_state.batch_page-=1
                    st.rerun()
            with page_col:
                st.markdown(
                    f'<div style="text-align:center;padding:.62rem 0;color:#999">Page '
                    f'<strong style="color:#fff">{st.session_state.batch_page}</strong> of {pages}</div>',
                    unsafe_allow_html=True,
                )
            with next_col:
                if st.button("Next →",disabled=next_disabled,use_container_width=True,key="next_page"):
                    st.session_state.batch_page+=1
                    st.rerun()

            page=st.session_state.batch_page
            start=(page-1)*page_size
            page_df=view.iloc[start:start+page_size].copy()
            display_cols=["Partner Term","Hiscox COB"] + (["Industry Code"] if show_code else []) + ["GL","PL","BOP","Cyber","Atlas Status","Action","Final COB"]
            final_options=sorted(set(cob_names + ["OOA"] + df["Final COB"].dropna().astype(str).tolist()))
            edited=st.data_editor(page_df[display_cols],use_container_width=True,hide_index=True,num_rows="fixed",column_config={
                "Partner Term":st.column_config.TextColumn(disabled=True,width="large"),
                "Hiscox COB":st.column_config.TextColumn(disabled=True,width="large"),
                "Industry Code":st.column_config.TextColumn(disabled=True,width="large"),
                "GL":st.column_config.TextColumn(disabled=True,width="small"),"PL":st.column_config.TextColumn(disabled=True,width="small"),"BOP":st.column_config.TextColumn(disabled=True,width="small"),"Cyber":st.column_config.TextColumn(disabled=True,width="small"),
                "Atlas Status":st.column_config.TextColumn(disabled=True,width="medium"),
                "Action":st.column_config.SelectboxColumn(options=["Keep Match","Change COB","Send to UWM"],width="medium",required=True),
                "Final COB":st.column_config.SelectboxColumn(options=final_options,width="large",required=True),
            },key=f"review_editor_{status_filter}_{page}_{show_code}")
            # Persist edits back to the canonical batch using the original page indexes.
            for idx in page_df.index:
                for col in ["Action","Final COB"]:
                    st.session_state.batch_results.at[idx,col]=edited.at[idx,col]
            shown_end=min(start+page_size,len(view))
            st.caption(f"Showing {start+1 if len(view) else 0} to {shown_end} of {len(view)} results")
            export=result_export(st.session_state.batch_results)
            st.download_button("DOWNLOAD RESULTS",export.to_csv(index=False),"atlas_results.csv","text/csv",use_container_width=False)
            st.markdown('</div>',unsafe_allow_html=True)
        footer()
