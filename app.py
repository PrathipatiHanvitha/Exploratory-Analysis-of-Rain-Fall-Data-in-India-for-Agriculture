from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/rainfall.pkl")
encoders = joblib.load("model/encoder.pkl")
impter = joblib.load("model/impter.pkl")
scaler = joblib.load("model/scale.pkl")

NUM_COLS = impter["num_cols"]
CAT_COLS = impter["cat_cols"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = dict(request.form)

    # 1) Build raw row
    row = {}

    # categorical: keep as string
    for col in CAT_COLS:
        v = form.get(col, "")
        row[col] = v if v != "" else np.nan

    # numeric: convert to float
    for col in NUM_COLS:
        v = form.get(col, "")
        try:
            row[col] = float(v)
        except:
            row[col] = np.nan

    # ✅ Auto date: today (and tomorrow)
    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    # If your model expects year/month/day, add them
    if "year" in NUM_COLS:
        row["year"] = float(today.year)
    if "month" in NUM_COLS:
        row["month"] = float(today.month)
    if "day" in NUM_COLS:
        row["day"] = float(today.day)

    X = pd.DataFrame([row])

    # 2) Impute missing
    X[NUM_COLS] = impter["num_imputer"].transform(X[NUM_COLS])
    X[CAT_COLS] = impter["cat_imputer"].transform(X[CAT_COLS])

    # 3) Encode categorical into numeric list (no pandas dtype issues)
    cat_encoded = []
    for col in CAT_COLS:
        le = encoders[col]
        val = str(X.loc[0, col])
        if val not in le.classes_:
            val = le.classes_[0]
        cat_encoded.append(int(le.transform([val])[0]))

    # 4) Combine into numeric feature array in correct order
    num_values = X.loc[0, NUM_COLS].astype(float).to_numpy()
    features = np.array(cat_encoded + num_values.tolist(), dtype=float).reshape(1, -1)

    # 5) Scale + Predict
    X_scaled = scaler.transform(features)
    pred = int(model.predict(X_scaled)[0])

    # Probability %
    if hasattr(model, "predict_proba"):
        prob_rain = float(model.predict_proba(X_scaled)[0][1])
    else:
        prob_rain = 0.5
    prob_percent = round(prob_rain * 100, 2)

    input_date_str = today.strftime("%d %b %Y")
    predicted_date_str = tomorrow.strftime("%d %b %Y")

    if pred == 1:
        return render_template(
            "chance.html",
            prob=prob_percent,
            input_date=input_date_str,
            predicted_date=predicted_date_str
        )

    return render_template(
        "noChance.html",
        prob=prob_percent,
        input_date=input_date_str,
        predicted_date=predicted_date_str
    )

if __name__ == "__main__":
    app.run(debug=True)