from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# CEO credentials
allowed_users = {
    "Samuel": "123",
    "Srini": "123",
    "Parath": "123"
}

# Load initial dataset
df = pd.read_csv("employee_data.csv")

# Load trained model + encoders
model = pickle.load(open("model.pkl", "rb"))
le_job = pickle.load(open("le_job.pkl", "rb"))
le_gender = pickle.load(open("le_gender.pkl", "rb"))

# Get dropdown options from dataset
job_roles = sorted(df["JobRole"].unique())
genders = sorted(df["Gender"].unique())
marital_statuses = sorted(df["MaritalStatus"].unique())

# ----------------- Routes -----------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if allowed_users.get(username) == password:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", message="Invalid username or password")
    return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    stay_count = len(df[df["Attrition"] == 0])
    leave_count = len(df[df["Attrition"] == 1])

    if request.method == "POST" and "age" in request.form:
        # Collect new employee data
        age = int(request.form["age"])
        income = float(request.form["income"])
        jobrole = request.form["jobrole"]
        years = int(request.form["years"])
        marital = int(request.form["marital"])
        gender = request.form["gender"]

        # Encode categorical features
        job_encoded = le_job.transform([jobrole])[0]
        gender_encoded = le_gender.transform([gender])[0]

        features = [[age, income, job_encoded, years, marital, gender_encoded]]
        pred = model.predict(features)[0]
        prediction = "Stay" if pred == 0 else "Leave"

        employee = {
            "EmployeeID": "NEW",
            "Age": age,
            "MonthlyIncome": income,
            "JobRole": jobrole,
            "YearsAtCompany": years,
            "MaritalStatus": marital,
            "Gender": gender
        }

        return render_template(
            "index.html",
            employee=employee,
            prediction=prediction,
            stay_count=stay_count,
            leave_count=leave_count,
            job_roles=job_roles,
            genders=genders,
            marital_statuses=marital_statuses
        )

    return render_template(
        "index.html",
        stay_count=stay_count,
        leave_count=leave_count,
        job_roles=job_roles,
        genders=genders,
        marital_statuses=marital_statuses
    )

@app.route("/search", methods=["POST"])
def search():
    if "user" not in session:
        return redirect(url_for("login"))

    emp_id = request.form.get("emp_id")
    employee = df[df["EmployeeID"] == emp_id]

    if employee.empty:
        stay_count = len(df[df["Attrition"] == 0])
        leave_count = len(df[df["Attrition"] == 1])
        return render_template(
            "index.html",
            message="❌ Employee ID not found",
            stay_count=stay_count,
            leave_count=leave_count,
            job_roles=job_roles,
            genders=genders,
            marital_statuses=marital_statuses
        )

    emp = employee.iloc[0].to_dict()
    job_encoded = le_job.transform([emp["JobRole"]])[0]
    gender_encoded = le_gender.transform([emp["Gender"]])[0]

    features = [[
        emp["Age"],
        emp["MonthlyIncome"],
        job_encoded,
        emp["YearsAtCompany"],
        emp["MaritalStatus"],
        gender_encoded
    ]]

    pred = model.predict(features)[0]
    prediction = "Stay" if pred == 0 else "Leave"

    stay_count = len(df[df["Attrition"] == 0])
    leave_count = len(df[df["Attrition"] == 1])

    return render_template(
        "index.html",
        employee=emp,
        prediction=prediction,
        stay_count=stay_count,
        leave_count=leave_count,
        job_roles=job_roles,
        genders=genders,
        marital_statuses=marital_statuses
    )

# ----------------- CSV Upload -----------------
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "user" not in session:
        return redirect(url_for("login"))

    file = request.files["file"]
    if file and file.filename.endswith(".csv"):
        global df
        df = pd.read_csv(file)
        message = f"✅ CSV '{file.filename}' uploaded successfully!"
    else:
        message = "❌ Invalid file! Please upload a CSV."

    stay_count = len(df[df["Attrition"] == 0])
    leave_count = len(df[df["Attrition"] == 1])

    return render_template(
        "index.html",
        message=message,
        stay_count=stay_count,
        leave_count=leave_count,
        job_roles=sorted(df["JobRole"].unique()),
        genders=sorted(df["Gender"].unique()),
        marital_statuses=sorted(df["MaritalStatus"].unique())
    )

# ----------------- Chart -----------------
@app.route("/chart")
def chart():
    if "user" not in session:
        return redirect(url_for("login"))

    stay_count = len(df[df["Attrition"] == 0])
    leave_count = len(df[df["Attrition"] == 1])

    return render_template(
        "chart.html",
        stay_count=stay_count,
        leave_count=leave_count
    )

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
