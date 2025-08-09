#!/usr/bin/env python3
"""
Build the AMBULATORY ML dashboard into ./ml/ so it won't replace your live index.html.
Outputs:
  ./ml/index.html
  ./ml/*.csv
  ./ml/*.png
Requires: models/model_amb.pkl (run scripts/train_ml_demo.py first)
"""
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(".", "ml")
MODEL_PATH = os.path.join(".", "models", "model_amb.pkl")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run scripts/train_ml_demo.py first (or include models/model_amb.pkl).")
    return joblib.load(MODEL_PATH)

def simulate_new_month(n=700):
    np.random.seed(20250809)
    start_date = datetime(2025, 6, 1)
    dates = [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 60, n)]
    departments = ["Peds Primary Care","Peds Cardiology","Peds Pulmonology","Peds GI","Peds Neuro"]
    providers = ["Dr. Shah","Dr. Kim","Dr. Rivera","PA Jordan","NP Miller"]
    tests = ["CBC","Lipid Panel","A1C","Throat Culture","Chest X-ray","CRP","Urinalysis"]
    diagnosis_groups = ["Well Child","Respiratory","Cardiac","Endocrine","Infectious","GI","Neuro"]
    follow_up_actions = ["Referral","Phone Outreach","Rx Started","Urgent Visit","Routine Follow-up","None"]
    df = pd.DataFrame({
        "patient_id": np.random.randint(10000, 50000, n),
        "visit_id": np.arange(1, n+1),
        "visit_date": dates,
        "department": np.random.choice(departments, n),
        "provider": np.random.choice(providers, n),
        "test_ordered": np.random.choice(tests, n),
        "diagnosis_group": np.random.choice(diagnosis_groups, n),
        "age": np.random.randint(1, 18, n),
        "outside_guideline": np.random.choice([0,1], n, p=[0.78,0.22]),
        "test_result": np.random.choice(["Normal","Abnormal","Missing"], n, p=[0.65,0.25,0.10]),
        "follow_up_action": np.random.choice(follow_up_actions, n, p=[0.10,0.18,0.12,0.05,0.20,0.35])
    })
    return df

def build_dashboard(df, model):
    # Predict
    cat = ["department","test_ordered","diagnosis_group","test_result","follow_up_action"]
    num = ["age","outside_guideline"]
    df["classification"] = model.predict(df[cat+num])

    # Aggregations
    dept_summary = (
        df.groupby(["department","classification"]).size()
          .reset_index(name="count").sort_values(["department","classification"])
    )
    dept_totals = dept_summary.groupby("department")["count"].transform("sum")
    dept_summary["percent"] = (dept_summary["count"]/dept_totals*100).round(1)
    dept_summary.to_csv(os.path.join(OUT_DIR,"summary_by_department_amb.csv"), index=False)

    dft = df.copy()
    dft["month"] = pd.to_datetime(dft["visit_date"]).dt.to_period("M").astype(str)
    non_act = (
        dft.assign(is_non=(dft["classification"]=="Non-actionable").astype(int))
           .groupby("month").agg(total_tests=("visit_id","count"),
                                 non_actionable=("is_non","sum")).reset_index()
    )
    non_act["non_actionable_rate"] = (non_act["non_actionable"]/non_act["total_tests"]*100).round(1)
    non_act.to_csv(os.path.join(OUT_DIR,"monthly_non_actionable_rate_amb.csv"), index=False)

    # Charts
    order=["Actionable","Non-actionable","Review","Incomplete"]
    pv=(dept_summary.pivot(index="department", columns="classification", values="percent").reindex(columns=order).fillna(0))
    import numpy as np
    fig1=plt.figure(figsize=(9,5)); bottom=np.zeros(len(pv)); x=np.arange(len(pv.index))
    for cls in order:
        vals=pv[cls].values if cls in pv else np.zeros(len(pv.index)); plt.bar(x, vals, bottom=bottom, label=cls); bottom+=vals
    plt.xticks(x, pv.index, rotation=20, ha="right"); plt.ylabel("Percent of Tests")
    plt.title("Ambulatory (ML): Classification by Department (Percent)")
    plt.legend(title="Classification", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"dept_classification_stacked_amb.png"), dpi=150); plt.close(fig1)

    fig2=plt.figure(figsize=(8,4.5))
    plt.plot(non_act["month"], non_act["non_actionable_rate"], marker="o")
    plt.xlabel("Month"); plt.ylabel("Non-actionable Rate (%)")
    plt.title("Ambulatory (ML): Monthly Non-actionable Diagnostic Test Rate")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"monthly_non_actionable_rate_amb.png"), dpi=150); plt.close(fig2)

    # Save data (ML outputs)
    df.to_csv(os.path.join(OUT_DIR,"ambulatory_tests_processed_amb.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR,"ambulatory_tests_raw_amb.csv"), index=False)

    # HTML page under /ml/
    html = f"""<!doctype html><html lang='en'><head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Ambulatory Diagnostic Stewardship – Dashboard (ML)</title>
<style>
:root {{ --bg:#f8fafc; --fg:#0f172a; --muted:#475569; --card:#ffffff; --accent:#1d4ed8; }}
html,body {{ margin:0; padding:0; background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }}
.container {{ max-width:1100px; margin:24px auto; padding:0 16px; }}
.card {{ background:var(--card); border-radius:16px; box-shadow:0 10px 20px rgba(0,0,0,0.05); padding:16px; }}
.grid {{ display:grid; gap:16px; grid-template-columns: 1fr; }}
@media (min-width: 900px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:0.9rem; margin-top:8px; }}
.badge {{ padding:2px 8px; border-radius:9999px; background:#e2e8f0; font-size:0.8rem; }}
.downloads a {{ display:inline-block; margin-right:12px; margin-bottom:8px; text-decoration:none; color: var(--accent); }}
</style></head><body>
<div class='container'>
  <div class='card' style='margin-bottom:16px;'>
    <div style='display:flex; gap:16px; flex-wrap:wrap; align-items:center;'>
      <div><span class='badge'>Model:</span> RandomForest (multiclass)</div>
      <div><span class='badge'>Rows:</span> {len(df)}</div>
      <div><span class='badge'>Path:</span> /ml/index.html</div>
    </div>
  </div>
  <div class='grid'>
    <div class='card'><figure><img src='dept_classification_stacked_amb.png' style='width:100%;border-radius:12px;'><figcaption>ML-predicted classification distribution by department.</figcaption></figure></div>
    <div class='card'><figure><img src='monthly_non_actionable_rate_amb.png' style='width:100%;border-radius:12px;'><figcaption>ML-predicted monthly non-actionable rate.</figcaption></figure></div>
  </div>
  <div class='card' style='margin-top:16px;'><h2 style='margin-top:0;'>Download Data</h2>
    <div class='downloads'>
      <a href='ambulatory_tests_raw_amb.csv' download>ambulatory_tests_raw_amb.csv</a>
      <a href='ambulatory_tests_processed_amb.csv' download>ambulatory_tests_processed_amb.csv</a>
      <a href='summary_by_department_amb.csv' download>summary_by_department_amb.csv</a>
      <a href='monthly_non_actionable_rate_amb.csv' download>monthly_non_actionable_rate_amb.csv</a>
    </div>
  </div>
  <div class='card' style='margin-top:16px;'><p class='subtitle'>This ML page is a side-by-side demo and does not replace your main dashboard.</p></div>
</div>
</body></html>
"""
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    model = ensure_model()
    df_new = simulate_new_month(n=700)
    build_dashboard(df_new, model)
    print("ML build completed at ./ml/index.html")

if __name__ == "__main__":
    main()
