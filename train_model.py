import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Pobierz dane z UCI (brak nagłówków, więc trzeba je podać ręcznie)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
    "Debtors", "ResidenceDuration", "Property", "Age", "OtherInstallmentPlans",
    "Housing", "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "Target"
]

df = pd.read_csv(url, sep=' ', header=None, names=columns)

# 2. Przetwarzanie danych
df['Target'] = df['Target'].map({1: 1, 2: 0})  # 1 = good, 2 = bad

# Przykład: wybierz kilka kolumn liczbowych + zakoduj kategorie
features = ["Duration", "CreditAmount", "InstallmentRate", "Age", "NumberCredits", "Job", "Housing"]
df = pd.get_dummies(df[features + ["Target"]], drop_first=True)

X = df.drop("Target", axis=1)
y = df["Target"]

# 3. Skalowanie
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 5. Zapisz model
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "credit_scaler.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("✅ Model został wytrenowany i zapisany.")