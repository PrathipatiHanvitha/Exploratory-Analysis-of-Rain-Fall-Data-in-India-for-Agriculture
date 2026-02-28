import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False


DATA_PATH = os.path.join("data", "weatherAUS.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading:", DATA_PATH)
data = pd.read_csv(DATA_PATH)

# ---- Date -> year/month/day ----
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["year"] = data["Date"].dt.year
data["month"] = data["Date"].dt.month
data["day"] = data["Date"].dt.day
data = data.drop(columns=["Date"])

# ---- Target ----
data = data.dropna(subset=["RainTomorrow"])
data["RainTomorrow"] = data["RainTomorrow"].map({"No": 0, "Yes": 1})

# ---- Keep columns similar to your project ----
wanted_cols = [
    "Location",
    "MinTemp", "MaxTemp", "Rainfall",
    "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm",
    "RainToday",
    "WindGustDir", "WindDir9am", "WindDir3pm",
    "year", "month", "day",
    "RainTomorrow"
]
wanted_cols = [c for c in wanted_cols if c in data.columns]
data = data[wanted_cols].copy()

# If RainToday exists as Yes/No, keep it categorical
if "RainToday" in data.columns:
    data["RainToday"] = data["RainToday"].fillna("No")

y = data["RainTomorrow"]
X = data.drop(columns=["RainTomorrow"])

# Force numeric conversion where possible
# Separate numeric and categorical columns properly
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical Columns:", cat_cols)
print("Numeric Columns:", num_cols)
# ---- Impute missing ----
num_imputer = SimpleImputer(strategy="mean")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

impter = {"num_imputer": num_imputer, "cat_imputer": cat_imputer, "num_cols": num_cols, "cat_cols": cat_cols}
joblib.dump(impter, os.path.join(MODEL_DIR, "impter.pkl"))

# ---- Encode categorical ----
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, os.path.join(MODEL_DIR, "encoder.pkl"))

# ---- Scale ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scale.pkl"))

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)

# ---- Models ----
models = {
    "rand_forest": RandomForestClassifier(n_estimators=150, random_state=0)
}

if HAS_XGB:
    models["xgboost"] = XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=0
    )

best_name, best_model, best_acc = None, None, -1

print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model

print("\nBest Model:", best_name, "Accuracy:", round(best_acc, 4))

# ---- Evaluation ----
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# ROC-AUC
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
else:
    y_prob = best_model.decision_function(X_test)
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 4))

# ---- Save best model ----
joblib.dump(best_model, os.path.join(MODEL_DIR, "rainfall.pkl"))

print("\nSaved files in /model:")
print(" - rainfall.pkl")
print(" - encoder.pkl")
print(" - impter.pkl")
print(" - scale.pkl")