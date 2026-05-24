# 🛡️ EFRMS — Enterprise Fraud Risk Management System

**RBI Master Directions Aligned | Built on PaySim Dataset**

---

## RBI Circulars Implemented

| Circular | Coverage in System |
|---|---|
| RBI/2016-17/344 (Master Directions on Frauds) | 8-category fraud classification, FMR-1/FMR-2 reporting |
| RBI/2019-20/170 | Customer zero-liability assessment per alert |
| DPSS.CO.PD No.1810/02.14.003/2015-16 | Large-value (≥₹1 Cr) flag, Flash report (≥₹50 Cr) |
| RBI/2022-23/75 | Digital payment fraud tagging |
| PMLA 2002, Sec 12 | STR filing checklist in Investigation Desk |

---

## Dashboard Modules

| Module | Purpose |
|---|---|
| 🏠 Command Centre | Live KPIs, priority alerts, RBI deadline countdown |
| ⚙️ Model Setup | Upload PaySim CSV or demo; trains exact notebook pipeline |
| 🔍 Transaction Screener | Real-time scoring; auto-RBI category + reporting obligation |
| 📁 Batch Upload | Bulk screen CSV; download results with RBI tags |
| 🚨 Alert Queue | Filter/assign/status-manage; large value + overdue flags |
| 🔬 Investigation Desk | Audit trail, notes, RBI compliance checklist, verdict, FMR number |
| 📊 Model Analytics | ROC, PR curves, confusion matrix, feature importance |
| 📋 RBI Compliance Hub | Circular reference, reporting framework, customer protection, stats |

---

## Alert Severity → RBI Reporting

| Severity | Fraud Prob | Action |
|---|---|---|
| CRITICAL | ≥ 85% | Immediate escalation; if ≥₹1 Cr → FMR-1 in 7 WD |
| HIGH | ≥ 60% | Under review within same day |
| MEDIUM | ≥ 35% | Review within 3 WD |
| LOW | < 35% | Informational; monitor |

---

## Model Pipeline (exact notebook)

```
PaySim CSV
  → dropna
  → drop nameOrig, nameDest
  → LabelEncoder (type)
  → Feature Engineering:
      amount_ratio          = amount / (oldbalanceOrg + 1)
      OriginalBalanceWError = oldbalanceOrg - amount - newbalanceOrig
      DestBalanceError      = oldbalanceDest + amount - newbalanceDest
      IsFullTransfer        = (newbalanceOrig == 0)
  → StandardScaler
  → train_test_split (80/20, stratify, random_state=42)
  → RandomForestClassifier (n=100, class_weight='balanced', random_state=42)
  → Evaluation: ROC-AUC, PR-AUC, Confusion Matrix
  → Threshold Tuning: best F1 (not fixed 0.5)
  → IsolationForest (n=300, contamination=0.005)
```

---

## Deploy (Free, Public URL)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "EFRMS"
git remote add origin https://github.com/YOUR_USERNAME/efrms.git
git push -u origin main

# 2. Go to share.streamlit.io → New app → repo → app.py → Deploy
```

Local run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset
[PaySim — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
`PS_20174392719_1491204439457_log.csv`

---

## Disclaimer
This is a prototype for educational/research purposes. In production, FMR filing must be done through the official RBI XBRL portal. STR/SAR must be filed with FIU-IND via the FINnet portal.
