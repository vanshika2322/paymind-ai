import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("payment_data.csv")

encoders = {}

for col in ["payment_method", "bank_status", "network_quality"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("success", axis=1)
y = df["success"]

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/payment_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")

print("Model trained successfully!")