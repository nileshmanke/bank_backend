# Bank Loan Risk Prediction - Flask API

This API serves the Random Forest model from your notebook's final prediction section.

## Model
- Dataset: 30,000 rows
- Algorithm: RandomForestClassifier
- Trees: 200
- Input features: Age, Annual_Income, Credit_Score, Debt_to_Income_Ratio, Loan_Amount, Loan_Term_Months, Past_Loans, Defaults
- Target: `Risk_Flag` (0 = LOW RISK, 1 = HIGH RISK)
- Holdout test accuracy from this training run: 97.83%

## Run on Windows

```bash
cd bank_loan_flask_api
py -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

The API runs at `http://127.0.0.1:5000`.

## Endpoints

- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`

## React example

```javascript
const response = await fetch("http://127.0.0.1:5000/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    Age: 39,
    Annual_Income: 412861,
    Credit_Score: 731,
    Debt_to_Income_Ratio: 0.18,
    Loan_Amount: 67047,
    Loan_Term_Months: 60,
    Past_Loans: 2,
    Defaults: 0
  })
});

const result = await response.json();
console.log(result);
```

Example successful response:

```json
{
  "success": true,
  "prediction": 0,
  "risk": "LOW RISK",
  "high_risk_probability": 0.12,
  "message": "Customer is classified as LOW RISK."
}
```

The probability above is illustrative; the API returns the actual model probability.

## Retrain the model

Put `advanced_bank_loan_dataset_30000.csv` in this folder and run:

```bash
python train_model.py
```

Then restart Flask.

## Important
This is an educational machine-learning classification demo, not a validated lending or credit decision system. Do not use its output alone to approve or reject real loan applications. The dataset's patterns, fairness, calibration, and performance on real customers have not been independently validated.
