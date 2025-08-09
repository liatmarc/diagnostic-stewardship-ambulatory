#!/usr/bin/env python3
"""
Train a demo ML model for ambulatory diagnostic stewardship classification.
- Simulates labeled data using a hidden rule
- Trains a RandomForestClassifier
- Saves model to models/model_amb.pkl
"""
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

OUT_DIR = "."
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Simulate labeled data
np.random.seed(123)
n = 2500
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 180, n)]

departments = ["Peds Primary Care","Peds Cardiology","Peds Pulmonology","Peds GI","Peds Neuro"]
tests = ["CBC","Lipid Panel","A1C","Throat Culture","Chest X-ray","CRP","Urinalysis"]
diagnosis_groups = ["Well Child","Respiratory","Cardiac","Endocrine","Infectious","GI","Neuro"]
follow_up_actions = ["Referral","Phone Outreach","Rx Started","Urgent Visit","Routine Follow-up","None"]

df = pd.DataFrame({
    "department": np.random.choice(departments, n),
    "test_ordered": np.random.choice(tests, n),
    "diagnosis_group": np.random.choice(diagnosis_groups, n),
    "test_result": np.random.choice(["Normal","Abnormal","Missing"], n, p=[0.66,0.24,0.10]),
    "follow_up_action": np.random.choice(follow_up_actions, n, p=[0.10,0.18,0.12,0.05,0.20,0.35]),
    "age": np.random.randint(1, 18, n),
    "outside_guideline": np.random.choice([0,1], n, p=[0.76,0.24])
})

def true_label(row):
    if row["test_result"] == "Missing":
        return "Incomplete"
    if row["test_result"] == "Abnormal" and row["follow_up_action"] != "None":
        return "Actionable"
    if row["test_result"] == "Normal" and row["outside_guideline"] == 1 and row["follow_up_action"] == "None":
        return "Non-actionable"
    return "Review"

df["label"] = df.apply(true_label, axis=1)
df.to_csv(os.path.join(OUT_DIR, "ml_training_data_amb.csv"), index=False)

# Train/validate
train = df.sample(frac=0.8, random_state=42)
valid = df.drop(train.index)

cat = ["department","test_ordered","diagnosis_group","test_result","follow_up_action"]
num = ["age","outside_guideline"]
pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
model = Pipeline([("prep", pre), ("rf", RandomForestClassifier(n_estimators=160, random_state=42))])
model.fit(train[cat+num], train["label"])

# Eval
pred = model.predict(valid[cat+num])
rep = classification_report(valid["label"], pred, digits=3)
cm = confusion_matrix(valid["label"], pred, labels=["Actionable","Non-actionable","Review","Incomplete"])

with open(os.path.join(OUT_DIR, "ml_training_report_amb.txt"), "w") as f:
    f.write(rep)

fig = plt.figure(figsize=(5,4))
import numpy as np
im = plt.imshow(cm, cmap="Blues")
plt.xticks(range(4), ["Actionable","Non-actionable","Review","Incomplete"], rotation=20, ha="right")
plt.yticks(range(4), ["Actionable","Non-actionable","Review","Incomplete"])
for i in range(4):
    for j in range(4):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.title("Confusion Matrix (Validation)")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ml_confusion_matrix_amb.png"), dpi=150)
plt.close(fig)

# Save model
import joblib
joblib.dump(model, os.path.join(MODEL_DIR, "model_amb.pkl"))
print("Saved model to", os.path.join(MODEL_DIR, "model_amb.pkl"))
