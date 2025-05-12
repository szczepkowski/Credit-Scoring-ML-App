import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("credit_model.pkl")
scaler = joblib.load("credit_scaler.pkl")
model_features = joblib.load("model_features.pkl")

# Załaduj model (lub stwórz tymczasowy)
try:
    model = joblib.load("credit_model.pkl")
except:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "credit_model.pkl")

# Dane wejściowe
class CreditInput(BaseModel):
    Duration: int
    CreditAmount: int
    InstallmentRate: int
    Age: int
    NumberCredits: int
    Job_A172: int = 0
    Job_A173: int = 0
    Job_A174: int = 0
    Housing_A152: int = 0
    Housing_A153: int = 0

@app.post("/predict")
def predict(data: CreditInput):
    print(data)
    df = pd.DataFrame([data.dict()])
    missing_cols = set(model_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[model_features]
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    return {"score": int(prediction), "probability": round(probability, 2)}