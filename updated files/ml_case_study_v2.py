"""
Practical Engineering Case Study: Supervised Machine Learning Classification (v2)
Algorithms:
1. Support Vector Machines (SVM)
2. Naive Bayes (Gaussian Naive Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost

NEW FEATURES IN THIS VERSION
  A. Stratified 5-fold cross-validation (more reliable than one split on 30 records)
  B. ROC curves + AUC for all five models
  C. Failure PROBABILITY and risk level (LOW / MEDIUM / HIGH) for new machines
  D. Automatic best-model selection, saved to disk with joblib
  E. Results exported to CSV
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# ============================================================

data = [
    # ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk
    [1,  48, 1.2, 1480,  900, 0],
    [2,  51, 1.5, 1490, 1200, 0],
    [3,  55, 1.8, 1475, 1500, 0],
    [4,  58, 2.0, 1500, 1800, 0],
    [5,  60, 2.2, 1460, 2100, 0],
    [6,  53, 1.4, 1510, 1100, 0],
    [7,  62, 2.4, 1450, 2300, 0],
    [8,  57, 1.7, 1495, 1600, 0],
    [9,  64, 2.6, 1440, 2500, 0],
    [10, 50, 1.3, 1505, 1000, 0],
    [11, 66, 2.8, 1435, 2700, 0],
    [12, 59, 2.1, 1485, 1900, 0],
    [13, 63, 2.5, 1465, 2200, 0],
    [14, 56, 1.9, 1498, 1400, 0],
    [15, 61, 2.3, 1470, 2000, 0],
    [16, 82, 5.8, 1370, 5200, 1],
    [17, 88, 6.5, 1340, 6100, 1],
    [18, 91, 7.1, 1320, 6800, 1],
    [19, 78, 5.2, 1390, 4800, 1],
    [20, 95, 7.8, 1290, 7400, 1],
    [21, 85, 6.1, 1360, 5700, 1],
    [22, 99, 8.4, 1260, 8100, 1],
    [23, 76, 4.9, 1400, 4500, 1],
    [24, 90, 7.0, 1310, 7000, 1],
    [25, 87, 6.7, 1330, 6400, 1],
    [26, 102, 8.8, 1240, 8600, 1],
    [27, 80, 5.5, 1380, 5000, 1],
    [28, 93, 7.5, 1280, 7600, 1],
    [29, 84, 6.0, 1350, 5900, 1],
    [30, 97, 8.1, 1270, 7900, 1],
]

columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "Speed_RPM", "Operating_Hours", "Failure_Risk"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 MACHINE RECORDS")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "Failure risk"}))

feature_cols = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"]
X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", style="Failure_Risk", s=100
)
plt.title("Machine temperature vs. vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of engineering variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

# probability=True lets SVM output failure probabilities (NEW: needed for ROC + risk level).
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                       probability=True, random_state=42))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. NEW: STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================
# With only 30 records, one 70/30 split is very noisy. Cross-validation
# trains/tests 5 times on different folds and averages the result.

print("\n" + "=" * 65)
print("5-FOLD STRATIFIED CROSS-VALIDATION (all 30 records)")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]
cv_rows = []

for model_name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    cv_rows.append({
        "Algorithm": model_name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Accuracy Std": scores["test_accuracy"].std(),
        "CV Precision": scores["test_precision"].mean(),
        "CV Recall": scores["test_recall"].mean(),
        "CV F1": scores["test_f1"].mean(),
    })

cv_df = pd.DataFrame(cv_rows).set_index("Algorithm")
print(cv_df.round(3))

cv_df[["CV Accuracy", "CV Precision", "CV Recall", "CV F1"]].plot(
    kind="bar", figsize=(11, 6), ylim=(0, 1.05)
)
plt.title("Cross-validated performance of five algorithms")
plt.ylabel("Mean score (5 folds)")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 5. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
roc_data = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # NEW: probability of failure for the ROC curve / AUC.
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[model_name] = (fpr, tpr, auc)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC": auc
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC      : {auc:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure risk"],
        zero_division=0
    ))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Failure risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 6. COMPARE ALGORITHMS + NEW ROC CURVES
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON (single train/test split)")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# NEW: ROC curves of all models on one plot.
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC curves: failure-risk detection")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering feature")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from the training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 8. NEW: SELECT BEST MODEL, SAVE IT, EXPORT RESULTS
# ============================================================

best_name = cv_df["CV F1"].idxmax()
print(f"\nBest model by cross-validated F1: {best_name} "
      f"(CV F1 = {cv_df.loc[best_name, 'CV F1']:.3f})")

# Refit the winner on ALL 30 records before saving it for use.
best_model = models[best_name]
best_model.fit(X, y)
joblib.dump(best_model, "best_failure_model.joblib")
print("Saved model  -> best_failure_model.joblib")

results_df.join(cv_df).round(3).to_csv("model_results.csv")
print("Saved results -> model_results.csv")
print("(Load later with: model = joblib.load('best_failure_model.joblib'))")

# ============================================================
# PART 9. NEW: PREDICT MULTIPLE MACHINES WITH RISK LEVELS
# ============================================================
# Illustrative values only, not real machine diagnoses.

def risk_level(p):
    """Convert failure probability into a maintenance category."""
    if p < 0.30:
        return "LOW"
    elif p < 0.70:
        return "MEDIUM"
    return "HIGH"

new_machines = pd.DataFrame([
    {"Temperature_C": 52, "Vibration_mm_s": 1.4, "Speed_RPM": 1500, "Operating_Hours": 1100},
    {"Temperature_C": 72, "Vibration_mm_s": 4.0, "Speed_RPM": 1410, "Operating_Hours": 3800},
    {"Temperature_C": 89, "Vibration_mm_s": 6.8, "Speed_RPM": 1325, "Operating_Hours": 6500},
])
new_machines.index = [f"Machine_{i}" for i in range(1, len(new_machines) + 1)]

print("\nNEW MACHINE OBSERVATIONS")
print(new_machines)

# Models were refit on all data only for the best one; refit the rest too so
# every algorithm predicts with the full dataset.
for model in models.values():
    model.fit(X, y)

for machine_id, row in new_machines.iterrows():
    sample = row.to_frame().T[feature_cols]
    print(f"\n{machine_id}:")
    for name, model in models.items():
        p_fail = model.predict_proba(sample)[0, 1]
        label = "Failure risk" if p_fail >= 0.5 else "Normal"
        print(f"  {name:15s} -> {label:12s} | P(failure) = {p_fail:5.1%} | Risk level: {risk_level(p_fail)}")

# Probability chart for the new machines (average across the five models).
avg_prob = pd.DataFrame({
    name: model.predict_proba(new_machines[feature_cols])[:, 1]
    for name, model in models.items()
}, index=new_machines.index)

avg_prob.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))
plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
plt.title("Predicted probability of failure for new machines")
plt.ylabel("P(failure)")
plt.xticks(rotation=0)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

print("\nEND OF PRACTICAL CASE STUDY")