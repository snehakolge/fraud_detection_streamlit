"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EFRMS — Enterprise Fraud Risk Management System                            ║
║  Built on PaySim dataset  |  Aligned with RBI Circular Guidelines           ║
║                                                                              ║
║  RBI Circulars Referenced:                                                   ║
║  • RBI/2017-18/15  – Fraud Classification & Reporting (Master Directions)   ║
║  • RBI/2019-20/170 – Customer Protection / Zero Liability                   ║
║  • DPSS.CO.PD No.1810/02.14.003/2015-16 – Large Value Fraud Reporting       ║
║  • RBI/2022-23/75  – Digital Payment Security Controls                       ║
║  • FMR-1 / FMR-2 / FMR-3 Reporting Framework                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve, auc, average_precision_score, classification_report,
    ConfusionMatrixDisplay
)
import datetime, warnings, uuid, json
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EFRMS | RBI-Aligned Fraud Risk Management",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── layout ── */
.stApp                          { background:#07111f; color:#e2e8f0; }
[data-testid="stSidebar"]       { background:#0b1729; border-right:1px solid #1e3a5f; }
[data-testid="stSidebar"] *     { color:#cbd5e1 !important; }
[data-testid="stSidebar"] hr    { border-color:#1e3a5f; }
[data-testid="stSidebar"] .stButton button {
    background:#1e3a5f !important; color:#e2e8f0 !important;
    border:1px solid #3b82f6 !important; border-radius:8px !important;
    font-size:.8rem !important;
}

/* ── KPI card ── */
.kpi { background:linear-gradient(135deg,#0f2744,#07111f);
       border:1px solid #1e3a5f; border-radius:14px;
       padding:1rem 1.3rem; text-align:center; }
.kpi-v { font-size:2.1rem; font-weight:700; margin:0; line-height:1.1; }
.kpi-l { font-size:.7rem; color:#64748b; margin:.3rem 0 0;
         text-transform:uppercase; letter-spacing:.08em; }

/* ── alert severity banners ── */
.ab { border-radius:10px; padding:.8rem 1rem;
      margin:.4rem 0; border-left:5px solid; }
.sCRITICAL { background:#3b0a0a; border-color:#ef4444; }
.sHIGH     { background:#3a1a05; border-color:#f97316; }
.sMEDIUM   { background:#2a2000; border-color:#eab308; }
.sLOW      { background:#052e16; border-color:#22c55e; }

/* ── status pills ── */
.pill { display:inline-block; padding:2px 9px; border-radius:20px;
        font-size:.72rem; font-weight:700; letter-spacing:.03em; }
.pOPEN       { background:#7f1d1d; color:#fca5a5; }
.pUNDER_REVIEW { background:#7c2d12; color:#fed7aa; }
.pESCALATED  { background:#2e1065; color:#c4b5fd; }
.pCLOSED_FP  { background:#14532d; color:#86efac; }
.pCLOSED_CF  { background:#1e1b4b; color:#a5b4fc; }
.pFMR_FILED  { background:#1e3a5f; color:#93c5fd; }

/* ── RBI info box ── */
.rbi-box { background:#0c1f38; border:1px solid #1d4ed8;
           border-left:4px solid #3b82f6;
           border-radius:8px; padding:.7rem 1rem;
           font-size:.82rem; margin:.5rem 0; }
.rbi-box a { color:#60a5fa; }

/* ── section label ── */
.sec { font-size:.78rem; font-weight:700; color:#64748b;
       text-transform:uppercase; letter-spacing:.1em;
       margin-bottom:.4rem; }

/* ── timer badge ── */
.timer-ok  { color:#22c55e; font-weight:700; }
.timer-warn{ color:#eab308; font-weight:700; }
.timer-over{ color:#ef4444; font-weight:700; }

h1,h2,h3 { color:#f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (RBI-aligned)
# ──────────────────────────────────────────────────────────────────────────────
TRANSACTION_TYPES = ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]
TYPE_MAP          = {t:i for i,t in enumerate(sorted(TRANSACTION_TYPES))}

# RBI Master Directions on Fraud – Fraud Category mapping
RBI_FRAUD_CATEGORIES = {
    "Misappropriation & Criminal Breach of Trust" : "MBT",
    "Fictitious Assets / Fraud Accounts"          : "FAF",
    "Cheating & Forgery"                          : "CNF",
    "Manipulation of Books"                       : "MOB",
    "Unauthorised Credit Facility"                : "UCF",
    "Card / Internet Fraud"                       : "CIF",
    "Cash Shortage"                               : "CSH",
    "Cybercrime / Digital Fraud"                  : "CYB",
    "Other Frauds"                                : "OTH",
}

# RBI Reporting timeline (working days after detection)
RBI_REPORTING_TIMELINE = {
    "≥ ₹1 Cr (FMR-1 to RBI CFRC)"     : 7,
    "≥ ₹50 Cr (Flash report within)"   : 1,
    "All others (FMR-2 internal)"       : 14,
}

# Severity thresholds (probability)
SEV_THRESHOLDS = {"CRITICAL":.85, "HIGH":.60, "MEDIUM":.35, "LOW":0.0}

# RBI Large Value threshold (in INR)
RBI_LARGE_VALUE_INR = 10_000_000   # ₹1 Crore

STATUSES = ["OPEN","UNDER_REVIEW","ESCALATED","CLOSED_CF","CLOSED_FP","FMR_FILED"]


# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
def _init():
    defs = dict(
        alerts=[],
        model=None, scaler=None, le=None,
        metrics={}, raw_df=None,
        analyst="Analyst-1",
        train_key=0,
    )
    for k,v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def engineer(df):
    """Exact feature engineering from the notebook."""
    d = df.copy()
    d['amount_ratio']          = d['amount'] / (d['oldbalanceOrg'] + 1)
    d['OriginalBalanceWError'] = d['oldbalanceOrg'] - d['amount'] - d['newbalanceOrig']
    d['DestBalanceError']      = d['oldbalanceDest'] + d['amount'] - d['newbalanceDest']
    d['IsFullTransfer']        = (d['newbalanceOrig'] == 0).astype(int)
    return d

def get_severity(prob):
    for sev, thr in SEV_THRESHOLDS.items():
        if prob >= thr:
            return sev
    return "LOW"

def sev_color(sev):
    return {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}.get(sev,"#94a3b8")

def rbi_fraud_category(tx_type, prob):
    """Map transaction type + probability to RBI fraud category."""
    if tx_type in ("TRANSFER","CASH_OUT") and prob >= .85:
        return "Cybercrime / Digital Fraud"
    if tx_type in ("DEBIT","PAYMENT"):
        return "Card / Internet Fraud"
    if prob >= .60:
        return "Misappropriation & Criminal Breach of Trust"
    return "Other Frauds"

def rbi_reporting_due(amount_inr, detected_ts):
    """Return (deadline_label, days_remaining, overdue)."""
    detected = datetime.datetime.strptime(detected_ts, "%Y-%m-%d %H:%M:%S")
    if amount_inr >= 50_000_000:   # ₹50 Cr
        deadline = detected + datetime.timedelta(days=1)
        label = "Flash Report (1 WD)"
    elif amount_inr >= RBI_LARGE_VALUE_INR:   # ₹1 Cr
        deadline = detected + datetime.timedelta(days=7)
        label = "FMR-1 to RBI CFRC (7 WD)"
    else:
        deadline = detected + datetime.timedelta(days=14)
        label = "FMR-2 Internal (14 WD)"
    remaining = (deadline - datetime.datetime.now()).days
    return label, remaining, remaining < 0

def push_alert(txn_id, amount, tx_type, prob, sev, note="Auto-generated by EFRMS screener"):
    fraud_cat  = rbi_fraud_category(tx_type, prob)
    rbi_label, rbi_days, overdue = rbi_reporting_due(amount, now_ts())
    a = {
        "alert_id"    : f"ALT-{uuid.uuid4().hex[:8].upper()}",
        "txn_id"      : txn_id,
        "raised_at"   : now_ts(),
        "type"        : tx_type,
        "amount"      : amount,
        "fraud_prob"  : round(prob*100, 2),
        "severity"    : sev,
        "status"      : "OPEN",
        "assigned_to" : "",
        "fraud_cat"   : fraud_cat,
        "rbi_report"  : rbi_label,
        "rbi_days_rem": rbi_days,
        "rbi_overdue" : overdue,
        "large_value" : amount >= RBI_LARGE_VALUE_INR,
        "zero_liab_applicable": True,   # RBI/2019-20/170
        "fmr_number"  : "",
        "notes"       : [{"ts":now_ts(),"user":"System","text":note}],
    }
    st.session_state.alerts.insert(0, a)

def get_idx(alert_id):
    for i,a in enumerate(st.session_state.alerts):
        if a["alert_id"] == alert_id:
            return i
    return -1


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING  — exact pipeline from the notebook
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Random Forest — please wait…")
def train_model(key, df):
    d = df.copy()
    # notebook: drop name cols → label encode type
    for col in ["nameOrig","nameDest"]:
        if col in d.columns: d.drop(col, axis=1, inplace=True)
    le = LabelEncoder()
    d['type'] = le.fit_transform(d['type'])
    # feature engineering (notebook cells)
    d = engineer(d)
    d.dropna(inplace=True)

    y = d["isFraud"]
    x = d.drop(["isFraud","isFlaggedFraud"], axis=1, errors="ignore")
    feat_names = x.columns.tolist()

    # notebook: StandardScaler
    sc = StandardScaler()
    xs = sc.fit_transform(x)

    # notebook: train_test_split stratify
    xtrain,xtest,ytrain,ytest = train_test_split(
        xs, y, test_size=0.2, random_state=42, stratify=y
    )

    # notebook: RandomForestClassifier n_estimators=100 balanced
    rfc = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    )
    rfc.fit(xtrain, ytrain)

    # notebook metrics
    ypred   = rfc.predict(xtest)
    yscores = rfc.predict_proba(xtest)[:,1]

    # PR-AUC (notebook)
    precision, recall, thresholds = precision_recall_curve(ytest, yscores)
    pr_auc = auc(recall, precision)

    # ROC-AUC (notebook)
    roc_auc_val = roc_auc_score(ytest, yscores)
    fpr, tpr, _ = roc_curve(ytest, yscores)

    # Threshold tuning (notebook best-threshold)
    f1_scores = (2*precision*recall)/(precision+recall+1e-10)
    best_thr  = float(thresholds[np.argmax(f1_scores)])

    # Feature importance (notebook)
    fi = dict(zip(feat_names, rfc.feature_importances_))

    # Confusion matrix (notebook)
    cm = confusion_matrix(ytest, ypred).tolist()

    # IsolationForest (notebook: n=300, contamination=0.005)
    iso = IsolationForest(n_estimators=300, contamination=0.005,
                          max_samples='auto', random_state=42)
    iso.fit(xs)

    metrics = {
        "roc_auc"   : round(roc_auc_val, 4),
        "pr_auc"    : round(pr_auc, 4),
        "threshold" : round(best_thr, 4),
        "feat_names": feat_names,
        "feat_imp"  : fi,
        "fpr"       : fpr.tolist(), "tpr": tpr.tolist(),
        "prec"      : precision.tolist(), "rec": recall.tolist(),
        "cm"        : cm,
        "report"    : classification_report(ytest, ypred, output_dict=True),
    }
    return rfc, sc, le, iso, metrics


def predict_one(row_dict):
    """Score one transaction. Returns fraud probability."""
    m   = st.session_state
    enc = TYPE_MAP.get(row_dict.get("type","PAYMENT"), 0)
    base = {
        "step"          : row_dict.get("step", 1),
        "type"          : enc,
        "amount"        : row_dict["amount"],
        "oldbalanceOrg" : row_dict["oldbalanceOrg"],
        "newbalanceOrig": row_dict["newbalanceOrig"],
        "oldbalanceDest": row_dict["oldbalanceDest"],
        "newbalanceDest": row_dict["newbalanceDest"],
    }
    df_ = pd.DataFrame([base])
    df_ = engineer(df_)
    fn  = m.metrics["feat_names"]
    for c in fn:
        if c not in df_.columns: df_[c] = 0
    df_ = df_[fn]
    scaled = m.scaler.transform(df_)
    return float(m.model.predict_proba(scaled)[0][1])


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ EFRMS")
    st.caption("Enterprise Fraud Risk Management System")
    st.markdown('<p style="font-size:.7rem;color:#64748b">RBI Master Directions Aligned</p>',
                unsafe_allow_html=True)
    st.divider()

    st.session_state.analyst = st.text_input("👤 Analyst", value=st.session_state.analyst)

    st.divider()
    page = st.radio("", [
        "🏠  Command Centre",
        "⚙️   Model Setup",
        "🔍  Transaction Screener",
        "📁  Batch Upload",
        "🚨  Alert Queue",
        "🔬  Investigation Desk",
        "📊  Model Analytics",
        "📋  RBI Compliance Hub",
    ])
    st.divider()

    # Live stats
    a_all   = st.session_state.alerts
    open_   = [a for a in a_all if a["status"]=="OPEN"]
    crit_   = [a for a in a_all if a["severity"]=="CRITICAL" and a["status"]=="OPEN"]
    over_   = [a for a in a_all if a.get("rbi_overdue") and a["status"] not in ("CLOSED_CF","CLOSED_FP","FMR_FILED")]
    lv_     = [a for a in a_all if a.get("large_value") and a["status"]=="OPEN"]

    c1,c2 = st.columns(2)
    c1.metric("Open",    len(open_))
    c2.metric("Critical",len(crit_))
    c1.metric("Overdue", len(over_))
    c2.metric("Large Val",len(lv_))

    if over_:
        st.error(f"⚠️ {len(over_)} alert(s) past RBI deadline!")

    st.divider()
    if st.session_state.model:
        st.success("✅ Model Active")
        st.caption(f"ROC-AUC : **{st.session_state.metrics.get('roc_auc')}**")
        st.caption(f"PR-AUC  : **{st.session_state.metrics.get('pr_auc')}**")
        st.caption(f"Threshold: **{st.session_state.metrics.get('threshold')}**")
    else:
        st.warning("⚠️ No model — go to Model Setup")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — COMMAND CENTRE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Command Centre":
    st.title("🛡️ EFRMS — Command Centre")
    st.caption(f"Analyst: **{st.session_state.analyst}**  |  {now_ts()}  |  RBI Master Directions on Fraud Aligned")
    st.divider()

    a_all = st.session_state.alerts
    total   = len(a_all)
    open_n  = sum(1 for a in a_all if a["status"]=="OPEN")
    crit_n  = sum(1 for a in a_all if a["severity"]=="CRITICAL" and a["status"]=="OPEN")
    high_n  = sum(1 for a in a_all if a["severity"]=="HIGH"     and a["status"]=="OPEN")
    esc_n   = sum(1 for a in a_all if a["status"]=="ESCALATED")
    over_n  = sum(1 for a in a_all if a.get("rbi_overdue") and a["status"] not in ("CLOSED_CF","CLOSED_FP","FMR_FILED"))
    lv_n    = sum(1 for a in a_all if a.get("large_value") and a["status"]=="OPEN")
    fmr_n   = sum(1 for a in a_all if a["status"]=="FMR_FILED")

    # KPI row 1
    cols = st.columns(4)
    kpis = [("Total Alerts",total,"#94a3b8"),
            ("Open",        open_n,"#3b82f6"),
            ("Critical",    crit_n,"#ef4444"),
            ("High",        high_n,"#f97316")]
    for col,(lbl,val,clr) in zip(cols,kpis):
        col.markdown(f'<div class="kpi"><p class="kpi-v" style="color:{clr}">{val}</p>'
                     f'<p class="kpi-l">{lbl}</p></div>', unsafe_allow_html=True)

    # KPI row 2
    cols2 = st.columns(4)
    kpis2 = [("Escalated",  esc_n, "#a855f7"),
             ("RBI Overdue",over_n,"#ef4444"),
             ("Large Value",lv_n,  "#f59e0b"),
             ("FMR Filed",  fmr_n, "#22c55e")]
    for col,(lbl,val,clr) in zip(cols2,kpis2):
        col.markdown(f'<div class="kpi"><p class="kpi-v" style="color:{clr}">{val}</p>'
                     f'<p class="kpi-l">{lbl}</p></div>', unsafe_allow_html=True)

    st.divider()

    if not st.session_state.model:
        st.info("👆 Go to **⚙️ Model Setup** to load data and train the fraud detection model.")
        st.markdown("""
### EFRMS Modules
| Module | Description |
|---|---|
| ⚙️ Model Setup | Upload PaySim CSV or use demo; trains RF model exactly as per your notebook |
| 🔍 Transaction Screener | Real-time single transaction risk scoring with RBI category |
| 📁 Batch Upload | Screen a CSV; auto-raise alerts above chosen severity |
| 🚨 Alert Queue | Filter, assign, status-manage all alerts |
| 🔬 Investigation Desk | Full investigation workflow per alert — audit trail, verdict, FMR number |
| 📊 Model Analytics | ROC, PR-AUC, confusion matrix, feature importance |
| 📋 RBI Compliance Hub | Circular reference, reporting deadlines, zero liability tracker |
        """)
    else:
        col_l, col_r = st.columns([3,2])

        with col_l:
            # Severity chart
            st.markdown('<p class="sec">Alert Severity Breakdown</p>', unsafe_allow_html=True)
            if a_all:
                sc_df = pd.Series([a["severity"] for a in a_all]).value_counts()\
                          .reindex(["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0)
                fig = go.Figure(go.Bar(
                    x=sc_df.index, y=sc_df.values,
                    marker_color=["#ef4444","#f97316","#eab308","#22c55e"],
                    text=sc_df.values, textposition="outside",
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(11,23,41,1)",
                                   font_color="white", height=220,
                                   margin=dict(t=10,b=20,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

            # RBI fraud category breakdown
            st.markdown('<p class="sec">RBI Fraud Category Breakdown</p>', unsafe_allow_html=True)
            if a_all:
                cat_df = pd.Series([a["fraud_cat"] for a in a_all]).value_counts()
                fig2 = px.pie(values=cat_df.values, names=cat_df.index,
                               color_discrete_sequence=px.colors.sequential.Blues_r,
                               hole=.4)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                    height=230, margin=dict(t=10,b=10))
                st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            st.markdown('<p class="sec">🔴 Priority Alerts</p>', unsafe_allow_html=True)
            priority = [a for a in a_all if a["severity"] in ("CRITICAL","HIGH")
                        and a["status"] not in ("CLOSED_CF","CLOSED_FP")][:7]
            if not priority:
                st.info("No critical/high open alerts.")
            for a in priority:
                sev = a["severity"]
                ov  = "⏰ OVERDUE" if a.get("rbi_overdue") else ""
                lv  = "💰 LARGE VALUE" if a.get("large_value") else ""
                st.markdown(f"""
<div class="ab s{sev}">
  <span style="font-weight:700;color:{sev_color(sev)}">[{sev}]</span>
  &nbsp;<strong>{a['alert_id']}</strong>
  <span class="pill p{a['status']}">{a['status']}</span>
  {f'<span style="color:#ef4444;font-size:.75rem"> {ov}</span>' if ov else ""}
  {f'<span style="color:#f59e0b;font-size:.75rem"> {lv}</span>' if lv else ""}<br>
  <small>{a['type']} | ₹{a['amount']:,.0f} | {a['fraud_prob']}% fraud prob</small><br>
  <small style="color:#94a3b8">{a['raised_at']} | {a['fraud_cat']}</small>
</div>""", unsafe_allow_html=True)

            # RBI Deadline countdown
            st.divider()
            st.markdown('<p class="sec">⏰ Upcoming RBI Deadlines</p>', unsafe_allow_html=True)
            deadline_alerts = [a for a in a_all
                               if a["status"] not in ("CLOSED_CF","CLOSED_FP","FMR_FILED")]
            deadline_alerts.sort(key=lambda x: x.get("rbi_days_rem", 999))
            for a in deadline_alerts[:4]:
                days = a.get("rbi_days_rem",0)
                cls  = "timer-over" if days<0 else "timer-warn" if days<=2 else "timer-ok"
                st.markdown(f"""
<div style="background:#0f2744;border-radius:8px;padding:.5rem .8rem;margin:.3rem 0;
            border-left:3px solid #1d4ed8">
  <small><strong>{a['alert_id']}</strong> — {a['rbi_report']}<br>
  Days remaining: <span class="{cls}">{days}d</span>
  | ₹{a['amount']:,.0f}</small>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️   Model Setup":
    st.title("⚙️ Model Setup")
    st.markdown("""
Upload the **PaySim CSV** (`PS_20174392719_1491204439457_log.csv`) — the same dataset
used in your notebook. The model pipeline follows your notebook exactly:

> Drop `nameOrig`/`nameDest` → LabelEncode `type` → Feature engineering
> → StandardScaler → train_test_split (stratify, 80/20) → RandomForestClassifier
> (n_estimators=100, class_weight='balanced') → PR-AUC + ROC-AUC + Best-Threshold (F1)
> → IsolationForest (n=300, contamination=0.005)
    """)

    uploaded = st.file_uploader("Upload PaySim CSV", type=["csv"])
    use_demo = st.checkbox("⚡ Use 10 000-row synthetic demo data (no file needed)")

    df_src = None
    if use_demo:
        np.random.seed(42)
        N = 10_000
        types  = np.random.choice(TRANSACTION_TYPES, N)
        amt    = np.random.exponential(55_000, N)
        old_o  = np.random.exponential(120_000, N)
        fraud  = np.random.choice([0,1], N, p=[.9913,.0087])
        new_o  = np.where(fraud, 0, np.maximum(old_o-amt, 0))
        old_d  = np.random.exponential(40_000, N)
        new_d  = np.where(fraud, old_d+amt*.9, old_d+amt)
        df_src = pd.DataFrame({
            "step":np.random.randint(1,744,N), "type":types, "amount":amt,
            "nameOrig":[f"C{i:07d}" for i in range(N)],
            "oldbalanceOrg":old_o, "newbalanceOrig":new_o,
            "nameDest":[f"M{i:07d}" for i in range(N)],
            "oldbalanceDest":old_d, "newbalanceDest":new_d,
            "isFraud":fraud, "isFlaggedFraud":0,
        })
        st.success(f"Demo: {N:,} rows | {int(fraud.sum())} fraud ({fraud.mean()*100:.2f}%)")
    elif uploaded:
        df_src = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_src):,} rows")

    if df_src is not None:
        st.session_state.raw_df = df_src
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Rows",     f"{len(df_src):,}")
        c2.metric("Fraud Cases",    f"{int(df_src['isFraud'].sum()):,}")
        c3.metric("Fraud Rate",     f"{df_src['isFraud'].mean()*100:.3f}%")

        with st.expander("Preview (first 10 rows)"):
            st.dataframe(df_src.head(10), use_container_width=True)

        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            key = id(df_src) + len(df_src)
            mdl, sc, le, iso, mets = train_model(key, df_src)
            st.session_state.model    = mdl
            st.session_state.scaler   = sc
            st.session_state.le       = le
            st.session_state.iso      = iso
            st.session_state.metrics  = mets
            st.session_state.train_key = key
            st.success(
                f"✅ Model trained!  ROC-AUC: **{mets['roc_auc']}** | "
                f"PR-AUC: **{mets['pr_auc']}** | Best Threshold: **{mets['threshold']}**"
            )
            st.markdown(f"""
<div class="rbi-box">
📌 <strong>RBI Note:</strong> Model uses best-threshold <strong>{mets['threshold']}</strong>
(tuned to maximise F1 on holdout set — consistent with RBI guidance on reducing false negatives
in high-value fraud detection).
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TRANSACTION SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Transaction Screener":
    st.title("🔍 Real-Time Transaction Screener")
    if not st.session_state.model:
        st.error("Train the model first in **⚙️ Model Setup**.")
        st.stop()

    st.markdown("Score a single transaction in real time. Alerts are raised automatically for scores above the F1-tuned threshold.")

    with st.form("screen"):
        c1,c2,c3 = st.columns(3)
        with c1:
            txn_id  = st.text_input("Transaction / UTR ID", value=f"UTR{uuid.uuid4().hex[:10].upper()}")
            tx_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
            amount  = st.number_input("Amount (₹)", min_value=0.01, value=75_000.0, step=500.0)
            step    = st.number_input("Step (Hour of simulation)", min_value=1, max_value=744, value=12)
        with c2:
            old_o = st.number_input("Sender Old Balance (₹)", min_value=0.0, value=1_20_000.0)
            new_o = st.number_input("Sender New Balance (₹)", min_value=0.0, value=45_000.0)
        with c3:
            old_d = st.number_input("Receiver Old Balance (₹)", min_value=0.0, value=0.0)
            new_d = st.number_input("Receiver New Balance (₹)", min_value=0.0, value=75_000.0)

        note = st.text_area("Analyst Note", placeholder="Optional context for this transaction…")
        submit = st.form_submit_button("🔍 Screen Transaction", type="primary", use_container_width=True)

    if submit:
        row  = {"step":step,"type":tx_type,"amount":amount,
                "oldbalanceOrg":old_o,"newbalanceOrig":new_o,
                "oldbalanceDest":old_d,"newbalanceDest":new_d}
        prob = predict_one(row)
        sev  = get_severity(prob)
        thr  = st.session_state.metrics["threshold"]
        flag = prob >= thr
        fcat = rbi_fraud_category(tx_type, prob)
        rlbl, rdays, rover = rbi_reporting_due(amount, now_ts())

        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Fraud Probability",f"{prob*100:.2f}%")
        c2.metric("Severity",          sev)
        c3.metric("Decision",         "🚨 FRAUD" if flag else "✅ LEGITIMATE")
        c4.metric("Model Threshold",   f"{thr*100:.1f}%")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob*100, 2),
            delta={"reference":thr*100,"suffix":"%","valueformat":".1f"},
            title={"text":"Fraud Risk Score (%)","font":{"color":"white"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"white"},
                "bar":{"color":sev_color(sev)},
                "steps":[
                    {"range":[0,35],"color":"#052e16"},
                    {"range":[35,60],"color":"#1c1917"},
                    {"range":[60,85],"color":"#431407"},
                    {"range":[85,100],"color":"#450a0a"},
                ],
                "threshold":{"line":{"color":"white","width":3},"value":thr*100},
            },
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # RBI context box
        bal_err = old_o - amount - new_o
        dest_err = old_d + amount - new_d
        st.markdown(f"""
<div class="rbi-box">
  📋 <strong>RBI Classification Context</strong><br>
  • <strong>Fraud Category:</strong> {fcat} ({RBI_FRAUD_CATEGORIES.get(fcat,'OTH')})<br>
  • <strong>RBI Reporting Obligation:</strong> {rlbl}
    {' <span style="color:#ef4444">⚠️ Would be OVERDUE</span>' if rover else ''}<br>
  • <strong>Zero Liability Applicable:</strong> Yes — RBI/2019-20/170 (if customer reported within 3 WD)<br>
  • <strong>Large Value Flag (≥₹1 Cr):</strong> {"🚨 YES — FMR-1 mandatory" if amount>=RBI_LARGE_VALUE_INR else "No"}<br>
  • <strong>Balance Error (Sender):</strong> ₹{bal_err:,.2f} {"⚠️ Inconsistency detected" if abs(bal_err)>1 else "✅ Consistent"}<br>
  • <strong>Balance Error (Receiver):</strong> ₹{dest_err:,.2f} {"⚠️ Inconsistency detected" if abs(dest_err)>1 else "✅ Consistent"}
</div>""", unsafe_allow_html=True)

        if flag:
            push_alert(txn_id, amount, tx_type, prob, sev, note or "Flagged by real-time screener.")
            st.markdown(f"""
<div class="ab s{sev}">
  🚨 <strong>FRAUD ALERT RAISED — {sev}</strong><br>
  Transaction <code>{txn_id}</code> scored <strong>{prob*100:.2f}%</strong>.
  Alert added to queue → go to <strong>🚨 Alert Queue</strong>.
</div>""", unsafe_allow_html=True)
        else:
            st.success(f"✅ Transaction {txn_id} appears legitimate ({prob*100:.2f}% < threshold {thr*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁  Batch Upload":
    st.title("📁 Batch Transaction Screening")
    if not st.session_state.model:
        st.error("Train the model first in **⚙️ Model Setup**.")
        st.stop()

    st.markdown("Upload a CSV of transactions. Columns: `step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`")

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
    use_demo_b = st.checkbox("⚡ Use 100-row synthetic demo batch")

    batch_df = None
    if use_demo_b:
        np.random.seed(9)
        n = 100
        batch_df = pd.DataFrame({
            "txn_id"       :[f"UTR{uuid.uuid4().hex[:8].upper()}" for _ in range(n)],
            "step"         :np.random.randint(1,744,n),
            "type"         :np.random.choice(TRANSACTION_TYPES, n),
            "amount"       :np.random.exponential(65_000,n),
            "oldbalanceOrg":np.random.exponential(110_000,n),
            "newbalanceOrig":np.random.exponential(50_000,n),
            "oldbalanceDest":np.random.exponential(30_000,n),
            "newbalanceDest":np.random.exponential(80_000,n),
        })
    elif batch_file:
        batch_df = pd.read_csv(batch_file)

    if batch_df is not None:
        st.dataframe(batch_df.head(5), use_container_width=True)

        min_sev = st.selectbox("Auto-raise alerts for severity ≥", ["CRITICAL","HIGH","MEDIUM","LOW"], index=1)
        SEV_RANK = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}

        if st.button("🚀 Run Batch Screening", type="primary", use_container_width=True):
            thr = st.session_state.metrics["threshold"]
            results, alerts_raised = [], 0
            prog = st.progress(0)
            for i, row in batch_df.iterrows():
                prob = predict_one({
                    "step":row.get("step",1), "type":row.get("type","PAYMENT"),
                    "amount":row.get("amount",0),
                    "oldbalanceOrg":row.get("oldbalanceOrg",0),
                    "newbalanceOrig":row.get("newbalanceOrig",0),
                    "oldbalanceDest":row.get("oldbalanceDest",0),
                    "newbalanceDest":row.get("newbalanceDest",0),
                })
                sev  = get_severity(prob)
                flag = prob >= thr
                txid = str(row.get("txn_id", f"TXN-{i+1:04d}"))
                amt  = float(row.get("amount",0))
                fcat = rbi_fraud_category(str(row.get("type","?")), prob)
                rlbl,_,_ = rbi_reporting_due(amt, now_ts())
                results.append({
                    "TXN ID"              : txid,
                    "Type"                : row.get("type","—"),
                    "Amount (₹)"          : round(amt, 2),
                    "Fraud Prob (%)"      : round(prob*100, 2),
                    "Severity"            : sev,
                    "Decision"            : "FRAUD" if flag else "SAFE",
                    "RBI Fraud Category"  : fcat,
                    "RBI Reporting"       : rlbl,
                    "Large Value"         : "YES" if amt>=RBI_LARGE_VALUE_INR else "No",
                })
                if flag and SEV_RANK[sev] <= SEV_RANK[min_sev]:
                    push_alert(txid, amt, str(row.get("type","?")), prob, sev,
                               "Auto-raised from batch screening.")
                    alerts_raised += 1
                prog.progress((i+1)/len(batch_df))

            prog.empty()
            res_df = pd.DataFrame(results)

            c1,c2,c3,c4 = st.columns(4)
            fraud_n = (res_df["Decision"]=="FRAUD").sum()
            lv_n    = (res_df["Large Value"]=="YES").sum()
            c1.metric("Screened",       len(res_df))
            c2.metric("Flagged Fraud",  fraud_n)
            c3.metric("Alerts Raised",  alerts_raised)
            c4.metric("Large Value (RBI)", lv_n)

            def row_color(row):
                if row["Decision"]=="FRAUD" and row["Large Value"]=="YES":
                    return ["background-color:#2e0a0a"]*len(row)
                if row["Decision"]=="FRAUD":
                    return ["background-color:#1a0808"]*len(row)
                return [""]*len(row)

            st.dataframe(res_df.style.apply(row_color,axis=1), use_container_width=True)
            st.download_button("⬇️ Download Results CSV", res_df.to_csv(index=False),
                                "batch_results.csv","text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ALERT QUEUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨  Alert Queue":
    st.title("🚨 Alert Queue")

    a_all = st.session_state.alerts
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total",         len(a_all))
    c2.metric("Open",          sum(1 for a in a_all if a["status"]=="OPEN"))
    c3.metric("In Review",     sum(1 for a in a_all if a["status"]=="UNDER_REVIEW"))
    c4.metric("Critical Open", sum(1 for a in a_all if a["severity"]=="CRITICAL" and a["status"]=="OPEN"))
    c5.metric("RBI Overdue",   sum(1 for a in a_all if a.get("rbi_overdue") and
                                    a["status"] not in ("CLOSED_CF","CLOSED_FP","FMR_FILED")))
    st.divider()

    # Filters
    fc1,fc2,fc3,fc4 = st.columns(4)
    f_sev    = fc1.multiselect("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"],
                                default=["CRITICAL","HIGH","MEDIUM","LOW"])
    f_status = fc2.multiselect("Status",   STATUSES, default=STATUSES)
    f_type   = fc3.multiselect("Tx Type",  TRANSACTION_TYPES, default=TRANSACTION_TYPES)
    f_lv     = fc4.checkbox("Large Value Only (≥₹1 Cr)")

    filtered = [
        a for a in a_all
        if a["severity"] in f_sev
        and a["status"]   in f_status
        and a["type"]     in f_type
        and (not f_lv or a.get("large_value"))
    ]
    # Sort: critical first, then by raised_at
    filtered.sort(key=lambda x: (STATUSES.index(x["status"]) if x["status"] in STATUSES else 99,
                                  SEV_THRESHOLDS.get(x["severity"],0)*-1))

    st.caption(f"Showing **{len(filtered)}** of **{len(a_all)}** alerts")
    st.divider()

    if not filtered:
        st.info("No alerts match selected filters.")
    else:
        for a in filtered:
            sev = a["severity"]
            idx = get_idx(a["alert_id"])
            ov_badge = ' <span style="color:#ef4444;font-size:.75rem">⏰ OVERDUE</span>' if a.get("rbi_overdue") else ""
            lv_badge = ' <span style="color:#f59e0b;font-size:.75rem">💰 LARGE VALUE</span>' if a.get("large_value") else ""

            col_l, col_r = st.columns([5,2])
            with col_l:
                st.markdown(f"""
<div class="ab s{sev}">
  <span style="font-weight:700;color:{sev_color(sev)}">[{sev}]</span>
  &nbsp;<strong>{a['alert_id']}</strong>&nbsp;
  <span class="pill p{a['status']}">{a['status']}</span>
  {ov_badge}{lv_badge}<br>
  <small><b>TXN:</b> {a['txn_id']} | <b>Type:</b> {a['type']}
  | <b>Amount:</b> ₹{a['amount']:,.0f}
  | <b>Fraud Prob:</b> {a['fraud_prob']}%
  | <b>Assigned:</b> {a['assigned_to'] or 'Unassigned'}</small><br>
  <small><b>RBI Category:</b> {a['fraud_cat']} | <b>Deadline:</b> {a['rbi_report']}</small><br>
  <small style="color:#94a3b8">Raised: {a['raised_at']}</small>
</div>""", unsafe_allow_html=True)

            with col_r:
                new_st = st.selectbox("Status", STATUSES,
                                       index=STATUSES.index(a["status"]) if a["status"] in STATUSES else 0,
                                       key=f"st_{a['alert_id']}")
                if new_st != a["status"]:
                    st.session_state.alerts[idx]["status"] = new_st
                    st.session_state.alerts[idx]["notes"].append({
                        "ts":now_ts(),"user":st.session_state.analyst,
                        "text":f"Status → {new_st}"})
                    st.rerun()

                new_assign = st.text_input("Assign to", value=a.get("assigned_to",""),
                                            key=f"asgn_{a['alert_id']}")
                if new_assign != a.get("assigned_to",""):
                    st.session_state.alerts[idx]["assigned_to"] = new_assign
                    st.rerun()

            st.divider()

    # Export
    if a_all:
        export_rows = [{k:v for k,v in a.items() if k!="notes"} for a in a_all]
        st.download_button("⬇️ Export All Alerts CSV",
                            pd.DataFrame(export_rows).to_csv(index=False),
                            "efrms_alerts.csv","text/csv", use_container_width=True)

    col_x,col_y = st.columns(2)
    if col_x.button("🧹 Clear Closed Alerts"):
        st.session_state.alerts = [a for a in st.session_state.alerts
                                    if a["status"] not in ("CLOSED_CF","CLOSED_FP")]
        st.rerun()
    if col_y.button("⚠️ Clear ALL Alerts"):
        st.session_state.alerts = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — INVESTIGATION DESK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Investigation Desk":
    st.title("🔬 Investigation Desk")
    a_all = st.session_state.alerts

    if not a_all:
        st.info("No alerts yet.")
        st.stop()

    chosen_id = st.selectbox("Select Alert to Investigate",
                              [a["alert_id"] for a in a_all])
    idx = get_idx(chosen_id)
    a   = st.session_state.alerts[idx]
    sev = a["severity"]

    st.divider()

    # ── Header ──
    ov_txt = "⏰ OVERDUE for RBI reporting" if a.get("rbi_overdue") else ""
    lv_txt = "💰 LARGE VALUE — FMR-1 mandatory" if a.get("large_value") else ""
    st.markdown(f"""
<div class="ab s{sev}" style="font-size:1rem">
  <span style="font-weight:800;color:{sev_color(sev)};font-size:1.3rem">[{sev}]</span>
  &nbsp;<strong>{a['alert_id']}</strong>
  <span class="pill p{a['status']}">{a['status']}</span>
  {f'<span style="color:#ef4444"> {ov_txt}</span>' if ov_txt else ""}
  {f'<span style="color:#f59e0b"> {lv_txt}</span>' if lv_txt else ""}
  <br><br>
  <table style="width:100%;color:#e2e8f0;font-size:.88rem">
    <tr>
      <td><b>TXN / UTR:</b> {a['txn_id']}</td>
      <td><b>Type:</b> {a['type']}</td>
      <td><b>Amount:</b> ₹{a['amount']:,.2f}</td>
    </tr>
    <tr>
      <td><b>Fraud Probability:</b> {a['fraud_prob']}%</td>
      <td><b>Raised:</b> {a['raised_at']}</td>
      <td><b>Assigned:</b> {a['assigned_to'] or 'Unassigned'}</td>
    </tr>
    <tr>
      <td><b>RBI Fraud Category:</b> {a['fraud_cat']}</td>
      <td><b>Reporting Deadline:</b> {a['rbi_report']}</td>
      <td><b>FMR Number:</b> {a.get('fmr_number') or 'Not filed'}</td>
    </tr>
  </table>
</div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns([3,2])

    # ── Left: audit trail + notes ──
    with col_l:
        st.subheader("📋 Audit Trail")
        for n in reversed(a["notes"]):
            st.markdown(f"""
<div style="border-left:3px solid #1e3a5f;padding:.45rem .8rem;
            margin:.35rem 0;background:#0f2744;border-radius:0 8px 8px 0">
  <small style="color:#94a3b8">{n['ts']} — <strong>{n['user']}</strong></small><br>
  <span style="font-size:.88rem">{n['text']}</span>
</div>""", unsafe_allow_html=True)

        st.subheader("✏️ Add Investigation Note")
        inv_note = st.text_area("Note", height=100,
                                 placeholder="Evidence found, customer contacted, SAR filed…")
        if st.button("💾 Save Note", use_container_width=True):
            if inv_note.strip():
                st.session_state.alerts[idx]["notes"].append({
                    "ts":now_ts(),"user":st.session_state.analyst,"text":inv_note.strip()})
                st.success("Note saved.")
                st.rerun()

        # RBI compliance section
        st.subheader("📋 RBI Compliance Checklist")
        checks = [
            ("Customer notified within 24 hrs (RBI/2019-20/170)", False),
            ("Zero liability assessment done",                     False),
            ("Internal FMR-2 raised",                             False),
            ("FMR-1 filed with RBI CFRC (if ≥ ₹1 Cr)",          False),
            ("SAR/STR submitted to FIU-IND (PMLA 2002, Sec 12)", False),
            ("Card / account blocked (if applicable)",            False),
            ("Law enforcement notified",                          False),
        ]
        check_state_key = f"checks_{a['alert_id']}"
        if check_state_key not in st.session_state:
            st.session_state[check_state_key] = [False]*len(checks)

        for ci,(label,_) in enumerate(checks):
            val = st.checkbox(label, value=st.session_state[check_state_key][ci],
                              key=f"chk_{a['alert_id']}_{ci}")
            st.session_state[check_state_key][ci] = val

    # ── Right: actions + gauge ──
    with col_r:
        st.subheader("🔧 Case Actions")

        new_st = st.selectbox("Update Status", STATUSES,
                               index=STATUSES.index(a["status"]) if a["status"] in STATUSES else 0,
                               key=f"inv_status_{a['alert_id']}")
        if new_st != a["status"] and st.button("Update Status"):
            st.session_state.alerts[idx]["status"] = new_st
            st.session_state.alerts[idx]["notes"].append({
                "ts":now_ts(),"user":st.session_state.analyst,"text":f"Status → {new_st}"})
            st.rerun()

        new_assign = st.text_input("Assign To", value=a.get("assigned_to",""),
                                    key=f"inv_assign_{a['alert_id']}")
        if st.button("Update Assignee"):
            st.session_state.alerts[idx]["assigned_to"] = new_assign
            st.session_state.alerts[idx]["notes"].append({
                "ts":now_ts(),"user":st.session_state.analyst,
                "text":f"Assigned to {new_assign}"})
            st.rerun()

        fmr_num = st.text_input("FMR Reference Number", value=a.get("fmr_number",""),
                                  placeholder="e.g. FMR-2025-XXXXXX")
        if st.button("Save FMR Number"):
            st.session_state.alerts[idx]["fmr_number"] = fmr_num
            st.session_state.alerts[idx]["notes"].append({
                "ts":now_ts(),"user":st.session_state.analyst,
                "text":f"FMR Number recorded: {fmr_num}"})
            st.rerun()

        verdict = st.radio("Verdict", ["—","Confirmed Fraud","False Positive","Needs Escalation"])
        if st.button("📝 Record Verdict", use_container_width=True):
            if verdict != "—":
                fs_map = {"Confirmed Fraud":"CLOSED_CF",
                          "False Positive":"CLOSED_FP",
                          "Needs Escalation":"ESCALATED"}
                st.session_state.alerts[idx]["status"] = fs_map[verdict]
                st.session_state.alerts[idx]["notes"].append({
                    "ts":now_ts(),"user":st.session_state.analyst,
                    "text":f"Verdict: {verdict} → {fs_map[verdict]}"})
                st.success(f"Verdict recorded.")
                st.rerun()

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=a["fraud_prob"],
            title={"text":"Fraud Score (%)","font":{"color":"white"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"white"},
                "bar":{"color":sev_color(sev)},
                "steps":[
                    {"range":[0,35],"color":"#052e16"},
                    {"range":[35,60],"color":"#1c1917"},
                    {"range":[60,85],"color":"#431407"},
                    {"range":[85,100],"color":"#450a0a"},
                ],
            },
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="white",height=240)
        st.plotly_chart(fig, use_container_width=True)

        # RBI deadline box
        days = a.get("rbi_days_rem",0)
        cls = "timer-over" if days<0 else "timer-warn" if days<=2 else "timer-ok"
        st.markdown(f"""
<div class="rbi-box">
  ⏰ <strong>RBI Reporting Deadline</strong><br>
  {a['rbi_report']}<br>
  Days remaining: <span class="{cls}"><strong>{days} days</strong></span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Analytics":
    st.title("📊 Model Analytics")
    if not st.session_state.model:
        st.error("Train the model first in **⚙️ Model Setup**.")
        st.stop()

    m = st.session_state.metrics

    c1,c2,c3 = st.columns(3)
    c1.metric("ROC-AUC",         m["roc_auc"])
    c2.metric("PR-AUC",          m["pr_auc"])
    c3.metric("Best Threshold",  m["threshold"])

    st.divider()
    t1,t2,t3,t4 = st.tabs(["ROC & PR Curves","Confusion Matrix","Feature Importance","Classification Report"])

    with t1:
        cl,cr = st.columns(2)
        with cl:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m["fpr"],y=m["tpr"],
                                      name=f'ROC AUC={m["roc_auc"]}',
                                      line=dict(color="#3b82f6",width=2)))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random",
                                      line=dict(dash="dash",color="#475569")))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(11,23,41,1)",font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=m["rec"],y=m["prec"],
                                       name=f'PR AP={m["pr_auc"]}',
                                       line=dict(color="#22c55e",width=2)))
            fig2.update_layout(title="Precision-Recall Curve",
                                xaxis_title="Recall", yaxis_title="Precision",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(11,23,41,1)",font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        cm = np.array(m["cm"])
        fig3 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                          x=["Pred Normal","Pred Fraud"],
                          y=["Actual Normal","Actual Fraud"])
        fig3.update_layout(title="Confusion Matrix",
                            paper_bgcolor="rgba(0,0,0,0)",font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

    with t3:
        fi_df = pd.DataFrame(list(m["feat_imp"].items()),
                              columns=["Feature","Importance"]).sort_values("Importance")
        fig4 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale="Blues",
                       title="Feature Importance")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(11,23,41,1)",
                            font_color="white", height=420)
        st.plotly_chart(fig4, use_container_width=True)

    with t4:
        rows=[]
        for label,vals in m["report"].items():
            if isinstance(vals,dict):
                rows.append({"Class":label,
                              "Precision":round(vals["precision"],4),
                              "Recall":round(vals["recall"],4),
                              "F1-Score":round(vals["f1-score"],4),
                              "Support":int(vals.get("support",0))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — RBI COMPLIANCE HUB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  RBI Compliance Hub":
    st.title("📋 RBI Compliance Hub")
    st.caption("Reference guide — RBI Master Directions on Fraud | Reporting Framework | Customer Protection")
    st.divider()

    tab1,tab2,tab3,tab4 = st.tabs([
        "📜 Key Circulars",
        "⏰ Reporting Obligations",
        "👤 Customer Protection",
        "📊 Alert Compliance Stats",
    ])

    with tab1:
        st.markdown("""
### Key RBI Circulars & Master Directions

<div class="rbi-box">
<strong>1. RBI Master Directions on Frauds – RBI/2016-17/344 (Updated)</strong><br>
Classification of frauds into 8 sub-categories. All scheduled commercial banks must report frauds
of ₹1 lakh and above to RBI using the FMR (Fraud Monitoring Return) framework.
</div>

<div class="rbi-box">
<strong>2. RBI/2019-20/170 — Customer Protection: Limiting Liability</strong><br>
Zero liability to customer if fraud is due to negligence of bank or third-party breach (reported within 3 working days).
Limited liability (₹5k–₹10k) applies for delay in reporting by customer.
</div>

<div class="rbi-box">
<strong>3. DPSS.CO.PD No.1810/02.14.003/2015-16 — Large Value Fraud Reporting</strong><br>
Frauds of ₹1 crore and above: FMR-1 must be filed with RBI CFRC within 7 working days of detection.
Frauds ≥ ₹50 crore: Flash report within 24 hours.
</div>

<div class="rbi-box">
<strong>4. RBI/2022-23/75 — Digital Payment Security Controls</strong><br>
Multi-factor authentication, transaction monitoring, real-time fraud detection for payment aggregators and banks.
</div>

<div class="rbi-box">
<strong>5. PMLA 2002 — Prevention of Money Laundering Act</strong><br>
Section 12: Reporting entities must file STR (Suspicious Transaction Report) with FIU-IND
within 7 days of forming suspicion.
</div>

<div class="rbi-box">
<strong>6. RBI/2021-22/119 — Mule Account Detection</strong><br>
Banks to use transaction monitoring systems (TMS) to detect and report mule/funnel accounts
involved in layering proceeds of crime.
</div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""### Fraud Reporting Obligations (FMR Framework)""")
        cols = st.columns(3)
        boxes = [
            ("FMR-1\nRBI CFRC Report",
             "Frauds ≥ ₹1 Crore",
             "File within **7 Working Days** of detection",
             "Online via XBRL portal to RBI CFRC\n(Central Fraud Registry Cell)",
             "#1d4ed8"),
            ("Flash Report\nImmediate Alert",
             "Frauds ≥ ₹50 Crore",
             "File within **24 Hours** of detection",
             "Direct communication to RBI Regional Office + CFRC",
             "#7c2d12"),
            ("FMR-2\nInternal & Quarterly",
             "All other frauds",
             "Internal report within **14 Working Days**\nQuarterly summary to Board",
             "Internal Risk Committee + Audit Committee of Board",
             "#14532d"),
        ]
        for col,(title,scope,deadline,channel,border) in zip(cols,boxes):
            col.markdown(f"""
<div style="background:#0f2744;border:1px solid {border};border-top:4px solid {border};
            border-radius:10px;padding:1rem;height:100%">
  <h4 style="color:#e2e8f0">{title}</h4>
  <p style="color:#94a3b8;font-size:.82rem"><b>Scope:</b> {scope}</p>
  <p style="color:#e2e8f0;font-size:.82rem">{deadline}</p>
  <p style="color:#94a3b8;font-size:.78rem">{channel}</p>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""
### RBI Fraud Classification (8 Sub-Categories)
| Code | Category | Typical Transactions |
|---|---|---|
| MBT | Misappropriation & Criminal Breach of Trust | Internal/staff fraud, embezzlement |
| FAF | Fictitious Assets / Fraud Accounts | Fictitious loan accounts, ghost accounts |
| CNF | Cheating & Forgery | Forged instruments, identity fraud |
| MOB | Manipulation of Books | Falsified ledger entries |
| UCF | Unauthorised Credit Facility | Unsanctioned loans/limits |
| CIF | Card / Internet Fraud | ATM, POS, online banking fraud |
| CSH | Cash Shortage | Teller fraud, vault shortage |
| CYB | Cybercrime / Digital Fraud | Phishing, account takeover, UPI fraud |
        """)

    with tab3:
        st.markdown("""### Customer Protection Framework — RBI/2019-20/170""")
        st.markdown("""
<div class="rbi-box">
<strong>Zero Liability — Customer NOT at fault</strong><br>
Full reimbursement within <strong>10 working days</strong> if:<br>
• Fraud due to negligence of bank/third party AND customer reports within 3 working days.<br>
• Unauthorised transaction from bank's system breach (regardless of reporting time).
</div>

<div class="rbi-box">
<strong>Limited Liability — Delay in Customer Reporting</strong><br>
| Delay | Max Customer Liability |<br>
|---|---|<br>
| 4–7 working days | ₹5,000 – ₹10,000 (depending on account type) |<br>
| > 7 working days | Board-approved policy applies |
</div>

<div class="rbi-box">
<strong>Bank Obligations</strong><br>
• Send SMS/email alert within 24 hrs of any debit transaction.<br>
• Provide 24×7 hotline to report fraud.<br>
• Resolve customer complaint and credit provisional amount within 10 WD.<br>
• Final resolution within 90 days.
</div>

<div class="rbi-box">
<strong>Digital Payments — Additional Controls (RBI/2022-23/75)</strong><br>
• MFA mandatory for all digital payments.<br>
• Real-time fraud scoring and transaction limits.<br>
• 2FA / OTP mandatory for transactions above ₹2,000 (UPI/IMPS).<br>
• Cool-off period for newly registered payees (₹2,000 limit for 24 hrs).
</div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### Alert Compliance Statistics")
        a_all = st.session_state.alerts
        if not a_all:
            st.info("No alerts generated yet. Use the Screener or Batch Upload to create alerts.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            total     = len(a_all)
            overdue   = sum(1 for a in a_all if a.get("rbi_overdue") and
                             a["status"] not in ("CLOSED_CF","CLOSED_FP","FMR_FILED"))
            lv_open   = sum(1 for a in a_all if a.get("large_value") and a["status"]=="OPEN")
            fmr_filed = sum(1 for a in a_all if a["status"]=="FMR_FILED")
            c1.metric("Total Alerts",   total)
            c2.metric("RBI Overdue ⚠️", overdue)
            c3.metric("Large Value Open",lv_open)
            c4.metric("FMR Filed ✅",   fmr_filed)

            # Category breakdown
            st.subheader("Alerts by RBI Fraud Category")
            cat_df = pd.Series([a["fraud_cat"] for a in a_all]).value_counts().reset_index()
            cat_df.columns = ["RBI Category","Count"]
            fig = px.bar(cat_df, x="RBI Category", y="Count",
                          color="Count", color_continuous_scale="Blues",
                          title="Alert Distribution by RBI Fraud Category")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(11,23,41,1)",font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            # Large value alerts
            lv_alerts = [a for a in a_all if a.get("large_value")]
            if lv_alerts:
                st.subheader(f"💰 Large Value Alerts (≥ ₹1 Cr) — {len(lv_alerts)} total")
                lv_df = pd.DataFrame([{
                    "Alert ID"  : a["alert_id"],
                    "TXN ID"    : a["txn_id"],
                    "Amount (₹)": f"{a['amount']:,.0f}",
                    "Severity"  : a["severity"],
                    "Status"    : a["status"],
                    "RBI Deadline": a["rbi_report"],
                    "Days Rem"  : a.get("rbi_days_rem","—"),
                    "Overdue"   : "YES ⚠️" if a.get("rbi_overdue") else "No",
                    "FMR Filed" : a.get("fmr_number") or "❌ Not filed",
                } for a in lv_alerts])
                st.dataframe(lv_df, use_container_width=True)
