import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("employee_data.csv")  # must contain EmployeeID column

# Encode categorical features
le_job = LabelEncoder()
df["JobRole"] = le_job.fit_transform(df["JobRole"])

le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

# Features and Target
X = df[["Age", "MonthlyIncome", "JobRole", "YearsAtCompany", "MaritalStatus", "Gender"]]
y = df["Attrition"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=500, solver="liblinear")
model.fit(X_train, y_train)

# Save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_job, open("le_job.pkl", "wb"))
pickle.dump(le_gender, open("le_gender.pkl", "wb"))

print("✅ Model, le_job.pkl, le_gender.pkl saved successfully!")
