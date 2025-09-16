import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = "please_change_this_secret"

DATA_PATH = "employee_data.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
LE_JOB_PATH = os.path.join(MODEL_DIR, "le_job.joblib")
LE_GENDER_PATH = os.path.join(MODEL_DIR, "le_gender.joblib")

# Simple credentials (change for real use)
allowed_users = {"Samuel":"123","Srini":"123","Parath":"123"}

# Load dataset (if exists)
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame(columns=["EmployeeID","Age","MonthlyIncome","JobRole","YearsAtCompany","Attrition","MaritalStatus","Gender"])

# Try load model + encoders
model = None
le_job = None
le_gender = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    le_job = joblib.load(LE_JOB_PATH)
    le_gender = joblib.load(LE_GENDER_PATH)
else:
    print("Model files not found. Run `python train_model.py` first to create model/ files.")

def encode_or_raise(enc, val, name):
    if enc is None:
        raise ValueError("Model / encoder not loaded. Run training.")
    if val not in enc.classes_:
        raise ValueError(f"Unknown {name} '{val}'. Update data and retrain model.")
    return int(enc.transform([val])[0])

def get_counts(df_local):
    stay = int(len(df_local[df_local["Attrition"] == 0]))
    leave = int(len(df_local[df_local["Attrition"] == 1]))
    return stay, leave

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")
        if allowed_users.get(u) == p:
            session["user"] = u
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/home", methods=["GET"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    stay_count, leave_count = get_counts(df)
    job_roles = sorted(df["JobRole"].dropna().unique().tolist())
    genders = sorted(df["Gender"].dropna().unique().tolist())
    marital_statuses = sorted(df["MaritalStatus"].dropna().unique().tolist())
    return render_template("home.html",
                           job_roles=job_roles,
                           genders=genders,
                           marital_statuses=marital_statuses,
                           stay_count=stay_count,
                           leave_count=leave_count)

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        age = int(request.form.get("age"))
        income = float(request.form.get("income"))
        jobrole = request.form.get("jobrole")
        years = int(request.form.get("years"))
        marital = int(request.form.get("marital"))
        gender = request.form.get("gender")

        job_enc = encode_or_raise(le_job, jobrole, "JobRole")
        gender_enc = encode_or_raise(le_gender, gender, "Gender")

        features = [[age, income, job_enc, years, marital, gender_enc]]

        if model is None:
            raise ValueError("Model not loaded; run training (train_model.py)")

        pred = int(model.predict(features)[0])
        prob = model.predict_proba(features)[0][1] * 100  # probability of leaving

        prediction_text = "Leave" if pred == 1 else "Stay"
        suggestion = []
        if prob > 70:
            suggestion.append("High attrition risk — review compensation and workload.")
        elif prob > 40:
            suggestion.append("Medium risk — consider career growth / mentoring.")
        else:
            suggestion.append("Low risk — maintain engagement.")

        employee = {
            "EmployeeID":"NEW",
            "Age": age,
            "MonthlyIncome": income,
            "JobRole": jobrole,
            "YearsAtCompany": years,
            "MaritalStatus": marital,
            "Gender": gender
        }

        stay_count, leave_count = get_counts(df)
        return render_template("result.html",
                               employee=employee,
                               prediction=prediction_text,
                               probability=round(prob,1),
                               suggestion=suggestion,
                               stay_count=stay_count,
                               leave_count=leave_count)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("home"))

@app.route("/search", methods=["POST"])
def search():
    if "user" not in session:
        return redirect(url_for("login"))
    emp_id = request.form.get("emp_id")
    emp = df[df["EmployeeID"] == emp_id]
    stay_count, leave_count = get_counts(df)
    if emp.empty:
        flash("Employee ID not found", "warning")
        return redirect(url_for("home"))
    emp = emp.iloc[0].to_dict()
    try:
        job_enc = encode_or_raise(le_job, emp["JobRole"], "JobRole")
        gender_enc = encode_or_raise(le_gender, emp["Gender"], "Gender")
        features = [[emp["Age"], emp["MonthlyIncome"], job_enc, emp["YearsAtCompany"], emp["MaritalStatus"], gender_enc]]
        pred = int(model.predict(features)[0])
        prob = model.predict_proba(features)[0][1] * 100
        prediction_text = "Leave" if pred == 1 else "Stay"
    except Exception as e:
        flash("Could not predict for this employee: " + str(e), "danger")
        prediction_text = None
        prob = None

    return render_template("home.html",
                           employee=emp,
                           prediction=prediction_text,
                           probability=round(prob,1) if prob is not None else None,
                           stay_count=stay_count,
                           leave_count=leave_count,
                           job_roles=sorted(df["JobRole"].unique().tolist()),
                           genders=sorted(df["Gender"].unique().tolist()),
                           marital_statuses=sorted(df["MaritalStatus"].unique().tolist()))

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "user" not in session:
        return redirect(url_for("login"))
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        flash("Please upload a CSV file", "danger")
        return redirect(url_for("home"))
    new_df = pd.read_csv(file)
    required = {"EmployeeID","Age","MonthlyIncome","JobRole","YearsAtCompany","Attrition","MaritalStatus","Gender"}
    if not required.issubset(set(new_df.columns)):
        flash("Uploaded CSV missing required columns", "danger")
        return redirect(url_for("home"))
    global df
    df = new_df.copy()
    flash(f"CSV uploaded ({file.filename}) — dashboard updated. Note: if CSV has new categories you should retrain the model.", "success")
    return redirect(url_for("home"))

@app.route("/chart")
def chart():
    if "user" not in session:
        return redirect(url_for("login"))
    stay_count, leave_count = get_counts(df)
    # Example: prepare attrition per job role
    job_counts = df.groupby("JobRole")["Attrition"].agg(['count','sum']).reset_index().to_dict(orient='records')
    return render_template("chart.html", stay_count=stay_count, leave_count=leave_count, job_counts=job_counts)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
