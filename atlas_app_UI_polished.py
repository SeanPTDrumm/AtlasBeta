import streamlit as st
import pandas as pd
import re
import html
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
# ---------------------------------------------------------------------------
# POLISHED ATLAS UI
# ---------------------------------------------------------------------------
RED="#E31B23"
st.markdown(f"""
<style>
:root{{--red:{RED};--ink:#171717;--muted:#6d6d6d;--line:#e5e5e5;--paper:#fff;--canvas:#f3f2f0}}
html,body,[class*="css"]{{font-family:Arial,Helvetica,sans-serif}}
.stApp{{background:var(--canvas);color:var(--ink)}}
[data-testid="stHeader"]{{background:transparent}} [data-testid="stToolbar"],#MainMenu,footer{{visibility:hidden}}
.block-container{{max-width:1160px;padding-top:1rem;padding-bottom:4rem}}
.hero{{background:linear-gradient(135deg,#050505 0%,#0c0c0c 72%,#211110 100%);border-radius:0 0 24px 24px;border-bottom:3px solid var(--red);box-shadow:0 18px 42px rgba(0,0,0,.14);padding:1.3rem 1.8rem;margin:-1rem 0 1.2rem}}
.kicker{{color:var(--red);font-size:.68rem;font-weight:850;letter-spacing:.18em;text-transform:uppercase}}
.hero-title{{color:#fff;font-size:1.55rem;font-weight:500;letter-spacing:-.02em;margin:.35rem 0}}
.hero-sub{{color:#bdbdbd;font-size:.87rem}}
div[role="radiogroup"]{{display:inline-flex;background:#fff;border:1px solid var(--line);border-radius:12px;padding:.3rem;box-shadow:0 5px 20px rgba(0,0,0,.04)}}
div[role="radiogroup"] label{{padding:.18rem .62rem;border-radius:8px;margin:0!important}}
div[role="radiogroup"] label:has(input:checked){{background:#111;color:#fff}}
[data-testid="stTextInput"] input{{background:#fff;border:1px solid #cfcfcf;border-radius:14px;min-height:56px;padding:0 1rem;font-size:1rem;box-shadow:0 6px 22px rgba(0,0,0,.04)}}
[data-testid="stTextInput"] input:focus{{border-color:var(--red);box-shadow:0 0 0 3px rgba(227,27,35,.1)}}
.stButton>button,.stDownloadButton>button{{border-radius:10px;border:1px solid #111;background:#111;color:#fff;min-height:43px;font-weight:750}}
.stButton>button:hover,.stDownloadButton>button:hover{{background:var(--red);border-color:var(--red);color:#fff}}
.section-kicker{{color:var(--red);font-size:.68rem;font-weight:850;letter-spacing:.16em;text-transform:uppercase}}
.section-title{{font-size:1.45rem;font-weight:650;letter-spacing:-.02em;margin:.2rem 0 .7rem}}
.micro{{color:var(--muted);font-size:.8rem}}
.card{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.5rem;box-shadow:0 10px 30px rgba(0,0,0,.055);margin:.8rem 0}}
.card-top{{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}}
.eyebrow{{color:var(--red);font-weight:850;letter-spacing:.14em;font-size:.67rem;text-transform:uppercase}}
.result-title{{font-size:1.85rem;line-height:1.15;font-weight:650;letter-spacing:-.025em;margin:.4rem 0 .15rem}}
.group{{color:var(--muted);font-size:.86rem}}
.pill{{display:inline-flex;align-items:center;gap:.45rem;padding:.5rem .8rem;border-radius:999px;font-size:.78rem;font-weight:850;white-space:nowrap}}
.dot{{width:8px;height:8px;border-radius:50%}}
.rec{{background:#e9f7ef;color:#17663b}} .rec .dot{{background:#2ba865}}
.conf{{background:#fff4d8;color:#7a5800}} .conf .dot{{background:#e0a400}}
.rev{{background:#fde8e8;color:#992727}} .rev .dot{{background:#d83c3c}}
.rule{{height:1px;background:var(--line);margin:1.2rem 0}}
.meta{{display:grid;grid-template-columns:1.05fr 1.55fr;gap:1.2rem}}
.label{{color:var(--muted);font-size:.67rem;font-weight:850;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.45rem}}
.code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92rem}}
.lobs{{display:grid;grid-template-columns:repeat(4,minmax(72px,1fr));gap:.55rem}}
.lob{{border:1px solid var(--line);border-radius:11px;padding:.65rem .7rem;display:flex;align-items:center;justify-content:space-between;background:#fafafa;font-size:.82rem;font-weight:750}}
.yes{{color:#1d7a48}} .no{{color:#8b8b8b}}
.definition{{font-size:.96rem;line-height:1.6;color:#353535;max-width:860px}}
[data-testid="stFileUploader"]{{background:#fff;border:1px dashed #bcbcbc;border-radius:16px;padding:.8rem}}
[data-testid="stMetric"]{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem;box-shadow:0 6px 20px rgba(0,0,0,.035)}}
[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:14px;overflow:hidden}}
[data-testid="stExpander"]{{background:#fff;border:1px solid var(--line);border-radius:12px}}
.preview-note{{background:#0b0b0b;color:#fff;border-left:4px solid var(--red);border-radius:10px;padding:.85rem 1rem;font-size:.85rem;margin:.8rem 0 1rem}}
@media(max-width:760px){{.block-container{{padding-left:1rem;padding-right:1rem}}.card-top{{display:block}}.pill{{margin-top:.8rem}}.meta{{grid-template-columns:1fr}}.lobs{{grid-template-columns:repeat(2,1fr)}}}}
</style>
""",unsafe_allow_html=True)

def safe(v): return html.escape(str(v if v is not None else ""))
def status_class(label): return "rec" if label=="Recommended" else ("conf" if label=="Please Confirm" else "rev")
def lob(name,v):
    ok=str(v).strip().lower() in ("yes","y","true")
    return f'<div class="lob"><span>{name}</span><span class="{"yes" if ok else "no"}">{"Yes" if ok else "No"}</span></div>'

def hero():
    st.markdown('<div class="hero">',unsafe_allow_html=True)
    a,b=st.columns([3.7,1.05],vertical_alignment="center")
    with a:
        st.markdown('<div class="kicker">Business classification</div><div class="hero-title">Find the right Hiscox classification and digital appetite.</div><div class="hero-sub">Search one business or map a complete partner list.</div>',unsafe_allow_html=True)
    with b:
        if Path("Atlas_Logo_A.png").exists(): st.image("Atlas_Logo_A.png",width=180)
        else: st.markdown('<div style="color:#fff;text-align:right;font-weight:800">HISCOX<br><span style="color:#E31B23">ATLAS</span></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

def result_card(result):
    _,label,help_text=user_status(result); attr=cob_attributes(result["Recommended_COB"])
    chips=''.join([lob("GL",attr["GL"]),lob("PL",attr["PL"]),lob("BOP",attr["BOP"]),lob("Cyber",attr["Cyber"])])
    code_label="Digital appetite" if attr.get("is_ooa") else "Industry code"
    code="Outside appetite" if attr.get("is_ooa") else safe(attr.get("Industry_Code","—"))
    group=f'<div class="group">{safe(attr.get("Group",""))}</div>' if attr.get("Group") else ""
    st.markdown(f"""<div class="card"><div class="card-top"><div><div class="eyebrow">Classification result</div><div class="result-title">{safe(attr['COB'])}</div>{group}</div><div class="pill {status_class(label)}"><span class="dot"></span>{safe(label)}</div></div><div class="rule"></div><div class="meta"><div><div class="label">{code_label}</div><div class="code">{code}</div></div><div><div class="label">Product availability</div><div class="lobs">{chips}</div></div></div><div class="rule"></div><div class="label">Definition</div><div class="definition">{safe(attr.get('Definition',''))}</div></div><div class="micro">{safe(help_text)}</div>""",unsafe_allow_html=True)

def details(result):
    with st.expander("Classification details"):
        a,b=st.columns(2)
        with a:
            st.caption("PIPELINE"); st.write(f"**Final stage:** {result.get('Final_Stage','')}"); st.write(f"**Review logic:** {result.get('Review_Confidence','')}")
            if result.get("Rule_Hit"): st.write(f"**Rule matched:** {result['Rule_Hit']}")
        with b:
            st.caption("MATCHING")
            if result.get("Top1_COB"): st.write(f"**Top match:** {result['Top1_COB']} ({result.get('Top1_Score','')})")
            if result.get("Top2_COB"): st.write(f"**Alternative:** {result['Top2_COB']}")
            if result.get("NAICS_COB"): st.write(f"**NAICS cross-check:** {result['NAICS_COB']} ({result.get('NAICS_Score','')})")

hero()
mode=st.radio("Workflow",["Search","Batch workspace"],horizontal=True,label_visibility="collapsed")
st.write("")
if mode=="Search":
    st.markdown('<div class="section-kicker">Single lookup</div><div class="section-title">What business are you classifying?</div>',unsafe_allow_html=True)
    query=st.text_input("Partner term or business description",placeholder="Try: fiber optic cable installation",label_visibility="collapsed")
    if query:
        r=run_pipeline(query); result_card(r); details(r)
else:
    st.markdown('<div class="section-kicker">Batch mapping</div><div class="section-title">Upload a partner list</div><div class="micro">CSV is supported in the current deployment.</div>',unsafe_allow_html=True)
    f=st.file_uploader("Upload CSV",type=["csv"],label_visibility="collapsed")
    names=[]
    if f is not None:
        df=pd.read_csv(f); a,b=st.columns([2,1])
        with a: st.markdown(f"**{safe(f.name)}**"); st.caption(f"{len(df):,} rows detected")
        with b: col=st.selectbox("Partner term column",df.columns.tolist())
        names=df[col].dropna().astype(str).tolist()
        with st.expander("Preview uploaded data"): st.dataframe(df.head(10),use_container_width=True,hide_index=True)
    if st.button("Classify partner list",type="primary") and names:
        if len(names)>300: st.warning(f"Only the first 300 of {len(names)} terms will be processed."); names=names[:300]
        bar=st.progress(0,text="Classifying partner terms..."); clean=[]; tech=[]
        for i,name in enumerate(names):
            r=run_pipeline(name); _,label,_=user_status(r); a=cob_attributes(r["Recommended_COB"])
            clean.append({"Partner Description":r["Partner_Name"],"Recommended COB":a["COB"],"Industry Code":a["Industry_Code"],"GL":"Y" if str(a["GL"]).lower() in ("yes","y","true") else "N","PL":"Y" if str(a["PL"]).lower() in ("yes","y","true") else "N","BOP":"Y" if str(a["BOP"]).lower() in ("yes","y","true") else "N","Cyber":"Y" if str(a["Cyber"]).lower() in ("yes","y","true") else "N","Status":label,"Review Action":"Accept" if label=="Recommended" else ("Confirm" if label=="Please Confirm" else "Leave for UWM review"),"Selected COB":a["COB"]})
            tech.append(r); bar.progress((i+1)/len(names),text=f"Classifying {i+1} of {len(names)}")
        bar.empty(); st.session_state.batch_clean=pd.DataFrame(clean); st.session_state.batch_tech=pd.DataFrame(tech)
    if "batch_clean" in st.session_state:
        df=st.session_state.batch_clean
        st.markdown('<div class="section-kicker" style="margin-top:1.4rem">Review workspace</div><div class="section-title">Classification results</div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4); m1.metric("Total",len(df)); m2.metric("Recommended",int((df.Status=="Recommended").sum())); m3.metric("Please Confirm",int((df.Status=="Please Confirm").sum())); m4.metric("Needs Review",int((df.Status=="Needs Review").sum()))
        filt=st.radio("Filter",["All","Recommended","Please Confirm","Needs Review"],horizontal=True)
        view=df if filt=="All" else df[df.Status==filt]
        st.markdown('<div class="preview-note"><strong>Review workspace preview</strong><br>Selections are session-only. The current CSV export remains unchanged while this workflow is tested.</div>',unsafe_allow_html=True)
        st.data_editor(view,use_container_width=True,hide_index=True,column_config={"Review Action":st.column_config.SelectboxColumn(options=["Accept","Confirm","Choose another COB","Leave for UWM review"]),"Selected COB":st.column_config.SelectboxColumn(options=sorted(cob_names)),"Partner Description":st.column_config.TextColumn(disabled=True),"Recommended COB":st.column_config.TextColumn(disabled=True),"Industry Code":st.column_config.TextColumn(disabled=True),"GL":st.column_config.TextColumn(disabled=True),"PL":st.column_config.TextColumn(disabled=True),"BOP":st.column_config.TextColumn(disabled=True),"Cyber":st.column_config.TextColumn(disabled=True),"Status":st.column_config.TextColumn(disabled=True)},key="atlas_review_editor")
        cols=["Partner Description","Recommended COB","Industry Code","GL","PL","BOP","Cyber","Status"]
        st.download_button("Download current Atlas results",df[cols].to_csv(index=False),"atlas_results.csv","text/csv")
        with st.expander("Underwriting / SME export"): st.download_button("Download technical results",st.session_state.batch_tech.to_csv(index=False),"atlas_results_technical.csv","text/csv")
