import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = "employee_data.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# REQUIRED COLUMNS:
expected = {"EmployeeID","Age","MonthlyIncome","JobRole","YearsAtCompany","Attrition","MaritalStatus","Gender"}
if not expected.issubset(set(df.columns)):
    raise SystemExit(f"CSV missing required columns. Found: {df.columns.tolist()}")

# Encode categorical features but keep original columns in df for dashboard
le_job = LabelEncoder()
le_gender = LabelEncoder()

df["JobRole_enc"] = le_job.fit_transform(df["JobRole"])
df["Gender_enc"]  = le_gender.fit_transform(df["Gender"])

X = df[["Age","MonthlyIncome","JobRole_enc","YearsAtCompany","MaritalStatus","Gender_enc"]]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
joblib.dump(le_job, os.path.join(MODEL_DIR, "le_job.joblib"))
joblib.dump(le_gender, os.path.join(MODEL_DIR, "le_gender.joblib"))

print("✅ Trained model and encoders saved to 'model/'")
