import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import os

DATASET_PATH = "advanced_bank_loan_dataset_30000.csv"
MODEL_PATH = "loan_risk_model.joblib"

FEATURES = [
    "Age", "Annual_Income", "Credit_Score", "Debt_to_Income_Ratio",
    "Loan_Amount", "Loan_Term_Months", "Past_Loans", "Defaults"
]
TARGET = "Risk_Flag"

df = pd.read_csv(DATASET_PATH)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

joblib.dump({"model": model, "features": FEATURES, "target": TARGET}, MODEL_PATH)

predictions = model.predict(X_test)
print("Model trained and saved:", MODEL_PATH)
print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions, zero_division=0))
