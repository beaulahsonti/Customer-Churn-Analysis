import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

print("Starting model training pipeline...")

# 1. Load the customer churn dataset
df = pd.read_csv("customer_churn_data.csv")
print(f"Loaded dataset: {len(df)} rows.")

# 2. Handle missing values in 'TotalCharges' by converting empty strings/whitespace to NaN and filling with the median
if "TotalCharges" in df.columns:
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    print(f"Preprocessed 'TotalCharges' column. Imputed missing values with median: ${median_val:.2f}")

# 3. Fit and Encode Categorical Features
categorical_cols = ["Gender", "Partner", "Dependents", "PhoneService", "InternetService", "ContractType"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on all categories
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"Encoded categorical column '{col}' with classes: {list(le.classes_)}")

# Encode the target label Churn (Yes=1, No=0)
churn_le = LabelEncoder()
df["Churn"] = churn_le.fit_transform(df["Churn"].astype(str))
encoders["Churn"] = churn_le
print(f"Encoded target column 'Churn' with classes: {list(churn_le.classes_)}")

# 4. Scale Numerical Features (Tenure, MonthlyCharges, TotalCharges)
numerical_cols = ["Tenure (months)", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("Scaled numerical features using StandardScaler.")

# 5. Prepare Features and Target Label
features = categorical_cols + numerical_cols
X = df[features]
y = df["Churn"]

# 6. Train the Neural Network Classifier (MLPClassifier)
print("Training MLPClassifier (Multi-Layer Perceptron)...")
model = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation="relu",
    solver="adam",
    max_iter=500, 
    random_state=42
)
model.fit(X, y)
print("MLPClassifier successfully trained!")

# 7. Save the trained model, categorical encoders, and scaler to the local directory
joblib.dump({"model": model, "encoders": encoders}, "mlp_churn_model.pkl")
joblib.dump(scaler, "churn_scaler.pkl")

print("Successfully exported artifacts:")
print(" - Model and categorical encoders saved to 'mlp_churn_model.pkl'")
print(" - Scaler saved to 'churn_scaler.pkl'")
