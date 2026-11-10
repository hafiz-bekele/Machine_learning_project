# ============================================================
# PUMP CONDITION CLASSIFICATION (v2)
# Six algorithms: KNN, SVM, Naive Bayes, Decision Tree,
#                 Random Forest, AdaBoost
#
# NEW FEATURES IN THIS VERSION
#   A. Pressure_bar is now used as an input feature (it was loaded but dropped)
#   B. New engineered feature: Stress_Index = Vibration x Temperature / 100
#   C. ROC curves + AUC for all six models (from cross-validated probabilities)
#   D. Hyperparameter tuning (GridSearchCV) for the best model
#   E. Failure PROBABILITY + risk level (LOW / MEDIUM / HIGH) for new pumps
#   F. Best model saved with joblib, results exported to CSV
# ============================================================

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_val_predict, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# display() only exists inside Jupyter/Colab. This makes it work in a normal .py file too.
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        # Styled tables keep their DataFrame in .data
        print(obj.data if hasattr(obj, "data") else obj)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET WITH COLUMN NAMES
# ============================================================

data = [
    [1, 1.2, 1500, 45, 8.5, 120, "Normal"],
    [2, 1.5, 1480, 48, 8.3, 250, "Normal"],
    [3, 1.8, 1520, 50, 8.6, 300, "Normal"],
    [4, 1.3, 1490, 46, 8.4, 180, "Normal"],
    [5, 2.0, 1510, 52, 8.2, 400, "Normal"],
    [6, 1.6, 1500, 49, 8.5, 220, "Normal"],
    [7, 2.1, 1470, 53, 8.1, 500, "Normal"],
    [8, 1.4, 1530, 47, 8.7, 150, "Normal"],
    [9, 2.2, 1490, 54, 8.3, 600, "Normal"],
    [10, 1.7, 1510, 50, 8.5, 350, "Normal"],
    [11, 2.5, 1450, 58, 7.9, 800, "Warning"],
    [12, 2.8, 1460, 60, 7.8, 950, "Warning"],
    [13, 3.0, 1440, 62, 7.7, 1100, "Warning"],
    [14, 3.2, 1430, 65, 7.5, 1250, "Warning"],
    [15, 2.7, 1470, 59, 7.9, 900, "Warning"],
    [16, 3.5, 1420, 67, 7.4, 1400, "Warning"],
    [17, 3.1, 1450, 63, 7.6, 1300, "Warning"],
    [18, 3.8, 1410, 69, 7.3, 1550, "Warning"],
    [19, 2.9, 1460, 61, 7.8, 1000, "Warning"],
    [20, 3.6, 1430, 68, 7.4, 1500, "Warning"],
    [21, 5.8, 1380, 78, 6.8, 2200, "Fault"],
    [22, 6.2, 1360, 82, 6.5, 2500, "Fault"],
    [23, 6.5, 1350, 85, 6.3, 2800, "Fault"],
    [24, 7.0, 1320, 89, 6.0, 3200, "Fault"],
    [25, 6.8, 1340, 87, 6.2, 3000, "Fault"],
    [26, 7.5, 1300, 92, 5.8, 3500, "Fault"],
    [27, 8.0, 1280, 95, 5.5, 3800, "Fault"],
    [28, 7.2, 1310, 90, 5.9, 3300, "Fault"],
    [29, 8.5, 1250, 98, 5.2, 4200, "Fault"],
    [30, 6.0, 1370, 80, 6.6, 2400, "Fault"],
    [31, 1.9, 1500, 51, 8.4, 380, "Normal"],
    [32, 2.3, 1480, 55, 8.0, 650, "Normal"],
    [33, 3.3, 1440, 64, 7.6, 1350, "Warning"],
    [34, 3.7, 1420, 70, 7.2, 1600, "Warning"],
    [35, 5.5, 1390, 76, 6.9, 2100, "Fault"],
    [36, 6.4, 1350, 84, 6.4, 2700, "Fault"],
    [37, 2.6, 1460, 57, 8.0, 850, "Warning"],
    [38, 1.8, 1520, 49, 8.6, 280, "Normal"],
    [39, 4.0, 1400, 72, 7.1, 1750, "Warning"],
    [40, 7.8, 1290, 94, 5.6, 3900, "Fault"]
]

# Each row is: ID, vibration, RPM, temperature, pressure, operating hours, label.
columns = ["ID", "Vibration_mm_s", "Speed_RPM", "Temperature_C",
           "Pressure_bar", "Operating_Hours", "Failure_Risk"]
df = pd.DataFrame(data, columns=columns)

# Map status classes into binary target classes (0 = Normal, 1 = High Risk/Warning/Fault)
df["Failure_Risk"] = df["Failure_Risk"].map({"Normal": 0, "Warning": 1, "Fault": 1})


# ------------------------------------------------------------
# NEW (B): feature engineering helper
# ------------------------------------------------------------
def add_engineered_features(frame):
    """Add Stress_Index: hot AND vibrating pumps score much higher than either alone."""
    frame = frame.copy()
    frame["Stress_Index"] = frame["Vibration_mm_s"] * frame["Temperature_C"] / 100
    return frame


df = add_engineered_features(df)

# NEW (A + B): Pressure_bar and Stress_Index are now part of the model inputs.
feature_cols = [
    "Temperature_C",
    "Vibration_mm_s",
    "Speed_RPM",
    "Operating_Hours",
    "Pressure_bar",
    "Stress_Index"
]

required_columns = ["ID"] + feature_cols + ["Failure_Risk"]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df[required_columns].copy()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
display(df.head())

print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "High Risk"}))

# Correlation heatmap (now includes the two new features)
plt.figure(figsize=(9, 6))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix (including new features)")
plt.tight_layout()
plt.show()


# ============================================================
# 4. FEATURE SELECTION
# ============================================================

X = df[feature_cols]
y = df["Failure_Risk"]

if y.isnull().any():
    raise ValueError("Target contains missing values.")

if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
    raise ValueError("Expected two target classes: 0 and 1.")

print("\nInput features:")
print(X.columns.tolist())
print("\nTarget variable:")
print("Failure_Risk")
print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTRAINING DATA")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTESTING DATA")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 6. DEFINE SIX CLASSIFICATION ALGORITHMS
# ============================================================
# NEW: SVM uses probability=True so it can output failure probabilities.

models = {

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=3))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                           probability=True, random_state=42))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
}

print("\nSix classification models created successfully!")


# ============================================================
# 7. MODEL TRAINING
# ============================================================

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained successfully!")

print("\nAll six models have been trained.")


# ============================================================
# 8. MODEL PREDICTIONS
# ============================================================

predictions = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    print(f"\n{name}")
    print("Predictions:", y_pred)
    print("Actual values:", y_test.to_numpy())


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

results = []

for name, y_pred in predictions.items():
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (High Risk)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (High Risk)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 (High Risk)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0)
    })

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
display(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision (High Risk)": "{:.3f}",
        "Recall (High Risk)": "{:.3f}",
        "F1 (High Risk)": "{:.3f}",
        "Macro F1": "{:.3f}"
    })
)


# ============================================================
# 10. CONFUSION MATRICES
# ============================================================

for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Normal", "High Risk"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    print(f"\n{name} Classification Report")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Normal", "High Risk"], zero_division=0
    ))


# ============================================================
# 11. VISUAL MODEL COMPARISON
# ============================================================

metrics = ["Accuracy", "Precision (High Risk)", "Recall (High Risk)", "Macro F1"]

for metric in metrics:
    plt.figure(figsize=(11, 5))
    sns.barplot(data=results_df, x="Algorithm", y=metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Machine Learning Algorithm")
    plt.ylabel(metric)
    plt.xticks(rotation=30)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


# ============================================================
# 12. FIVE-FOLD CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    cv_results.append({
        "Algorithm": name,
        "Mean Macro F1": scores.mean(),
        "Standard Deviation": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

print("\nCROSS-VALIDATION RESULTS")
display(cv_df.sort_values(by="Mean Macro F1", ascending=False))


# ============================================================
# 12B. NEW (C): ROC CURVES + AUC FOR ALL SIX MODELS
# ============================================================
# The test set has only 8 records, so a ROC curve on it would be very coarse.
# Instead, each of the 40 pumps gets a probability from a model that never saw it
# (cross_val_predict), which gives a much smoother and more honest curve.

print("\nROC / AUC (cross-validated probabilities, all 40 pumps)")

auc_scores = {}
plt.figure(figsize=(7, 6))

for name, model in models.items():
    proba = cross_val_predict(
        clone(model), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    auc_scores[name] = auc
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC curves: detecting high-risk pumps")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

auc_df = pd.DataFrame({"Algorithm": list(auc_scores.keys()),
                       "CV AUC": list(auc_scores.values())})
display(auc_df.sort_values("CV AUC", ascending=False))


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

tree_model = models["Decision Tree"]
forest_model = models["Random Forest"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Decision Tree": tree_model.feature_importances_,
    "Random Forest": forest_model.feature_importances_
})

print("\nFEATURE IMPORTANCE")
display(importance_df)

importance_df.set_index("Feature").plot(kind="bar", figsize=(10, 5))
plt.title("Feature Importance Comparison")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================================
# 14. MODEL RANKING
# ============================================================
# The best model is chosen with CROSS-VALIDATION on the training data,
# NOT with the test set (the test set stays unseen during model selection).

ranked_models = cv_df.sort_values(
    by=["Mean Macro F1", "Standard Deviation"],
    ascending=[False, True]
)

print("\nMODEL RANKING (by cross-validation Macro F1)")
display(ranked_models)

best_name = ranked_models.iloc[0]["Algorithm"]
print("\nBest model selected by cross-validation:", best_name)

best_model = models[best_name]

print("\nBest model classification report on the test set:")
print(classification_report(
    y_test, predictions[best_name],
    target_names=["Normal", "High Risk"], zero_division=0
))


# ============================================================
# 14B. NEW (D): HYPERPARAMETER TUNING OF THE BEST MODEL
# ============================================================
# GridSearchCV tries every parameter combination with 5-fold CV on the
# TRAINING data only, so the test set is still untouched.

param_grids = {
    "KNN": {
        "classifier__n_neighbors": [3, 5, 7],
        "classifier__weights": ["uniform", "distance"]
    },
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ["scale", 0.1, 0.01]
    },
    "Naive Bayes": {
        "classifier__var_smoothing": [1e-9, 1e-8, 1e-6]
    },
    "Decision Tree": {
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5]
    },
    "AdaBoost": {
        "n_estimators": [30, 50, 100],
        "learning_rate": [0.5, 1.0]
    }
}

grid = GridSearchCV(
    clone(best_model),
    param_grids[best_name],
    cv=cv,
    scoring="f1_macro"
)
grid.fit(X_train, y_train)

tuned_model = grid.best_estimator_

print(f"\nHYPERPARAMETER TUNING: {best_name}")
print("Best parameters :", grid.best_params_)
print(f"Best CV Macro F1: {grid.best_score_:.3f}")

before = f1_score(y_test, best_model.predict(X_test), average="macro", zero_division=0)
after = f1_score(y_test, tuned_model.predict(X_test), average="macro", zero_division=0)
print(f"Test Macro F1 before tuning: {before:.3f}")
print(f"Test Macro F1 after tuning : {after:.3f}")


# ============================================================
# 14C. NEW (F): SAVE THE FINAL MODEL AND THE RESULTS
# ============================================================

# Refit the tuned model on ALL 40 pumps before saving it for real use.
final_model = clone(tuned_model).fit(X, y)

joblib.dump(final_model, "best_pump_model.joblib")
print("\nSaved model   -> best_pump_model.joblib")

all_results = results_df.merge(cv_df, on="Algorithm").merge(auc_df, on="Algorithm")
all_results.round(3).to_csv("pump_model_results.csv", index=False)
print("Saved results -> pump_model_results.csv")
print("Load later with: model = joblib.load('best_pump_model.joblib')")


# ============================================================
# 15. NEW (E): NEW PUMP PREDICTIONS WITH PROBABILITY AND RISK LEVEL
# ============================================================
# Example values are illustrative, not real diagnoses.

def risk_level(p):
    """Convert the probability of high risk into a maintenance category."""
    if p < 0.30:
        return "LOW"
    elif p < 0.70:
        return "MEDIUM"
    return "HIGH"


new_pumps = pd.DataFrame([
    {"Temperature_C": 48, "Vibration_mm_s": 1.5, "Speed_RPM": 1500,
     "Operating_Hours": 250,  "Pressure_bar": 8.4},
    {"Temperature_C": 62, "Vibration_mm_s": 3.0, "Speed_RPM": 1440,
     "Operating_Hours": 1100, "Pressure_bar": 7.7},
    {"Temperature_C": 75, "Vibration_mm_s": 6.5, "Speed_RPM": 1450,
     "Operating_Hours": 5000, "Pressure_bar": 6.9},
    {"Temperature_C": 92, "Vibration_mm_s": 7.5, "Speed_RPM": 1300,
     "Operating_Hours": 3500, "Pressure_bar": 5.8},
])
new_pumps.index = [f"Pump_{i}" for i in range(1, len(new_pumps) + 1)]

# The engineered feature must be created for new data too.
new_pumps = add_engineered_features(new_pumps)[feature_cols]

print("\nNEW PUMP OBSERVATIONS")
display(new_pumps)

# Final (tuned) model: probability, class and risk level.
p_high = final_model.predict_proba(new_pumps)[:, 1]

report = pd.DataFrame({
    "P(High Risk)": p_high,
    "Prediction": np.where(p_high >= 0.5, "HIGH RISK", "NORMAL / LOW RISK"),
    "Risk Level": [risk_level(p) for p in p_high]
}, index=new_pumps.index)

print(f"\nFINAL MODEL ({best_name}, tuned) PREDICTIONS")
print(report.assign(**{"P(High Risk)": report["P(High Risk)"].map("{:.1%}".format)}))

# Probability from every model, for comparison.
prob_table = pd.DataFrame(
    {name: model.predict_proba(new_pumps)[:, 1] for name, model in models.items()},
    index=new_pumps.index
)

print("\nP(High Risk) from all six models (trained on the training set):")
print(prob_table.map("{:.1%}".format) if hasattr(prob_table, "map") else prob_table.applymap("{:.1%}".format))

prob_table.plot(kind="bar", figsize=(11, 5), ylim=(0, 1.05))
plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
plt.title("Predicted probability of high risk for new pumps")
plt.ylabel("P(High Risk)")
plt.xticks(rotation=0)
plt.legend(loc="upper left", ncol=2)
plt.tight_layout()
plt.show()

print("\nEND OF CASE STUDY")