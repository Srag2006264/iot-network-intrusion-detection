import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# N-BaIoT RANDOM FOREST EVALUATION
# ============================================================

print("=" * 70)
print("N-BaIoT RANDOM FOREST EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "model",
    "random_forest.joblib"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "evaluation"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "results.json"
)


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — CHECKING FILES")
print("=" * 70)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_FILE}"
    )

print("\n✓ Random Forest model found.")
print("✓ Test dataset found.")


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — LOADING MODEL")
print("=" * 70)

model = joblib.load(
    MODEL_FILE
)

print("\nRandom Forest loaded successfully.")


# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — LOADING TEST DATA")
print("=" * 70)

test_df = pd.read_csv(
    TEST_FILE
)

print("\nTest dataset loaded.")

print(
    f"Rows: {len(test_df):,}"
)

print(
    f"Columns: {len(test_df.columns)}"
)


# ------------------------------------------------------------
# GET FEATURES FROM MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — VERIFYING MODEL FEATURE SCHEMA")
print("=" * 70)

if not hasattr(
    model,
    "feature_names_in_"
):

    raise ValueError(
        "Model does not contain feature_names_in_."
    )


feature_columns = (
    model.feature_names_in_.tolist()
)


print(
    f"\nModel expects: "
    f"{len(feature_columns)} features"
)


if len(feature_columns) != 115:

    raise ValueError(
        "Model does not contain exactly 115 features."
    )


# ------------------------------------------------------------
# VERIFY TEST DATA SCHEMA
# ------------------------------------------------------------

test_feature_columns = [
    column
    for column in test_df.columns
    if column != "label"
]


if feature_columns != test_feature_columns:

    raise ValueError(
        "Test feature schema does not exactly "
        "match the trained model."
    )


print(
    "SUCCESS: Test features exactly match "
    "the trained model."
)


# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "label"
]


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — GENERATING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test
)

print(
    "\nPredictions generated successfully."
)


# ------------------------------------------------------------
# PREDICT PROBA
# ------------------------------------------------------------

predict_proba_supported = hasattr(
    model,
    "predict_proba"
)


if predict_proba_supported:

    y_probability = model.predict_proba(
        X_test
    )

    print(
        "predict_proba() is supported."
    )

else:

    y_probability = None

    print(
        "predict_proba() is not supported."
    )


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — CALCULATING METRICS")
print("=" * 70)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)


recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)


f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


print(
    f"\nAccuracy:  {accuracy:.10f}"
)

print(
    f"Precision: {precision:.10f}"
)

print(
    f"Recall:    {recall:.10f}"
)

print(
    f"F1-score:  {f1:.10f}"
)


# ------------------------------------------------------------
# CLASS-SPECIFIC METRICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal",
        "Attack"
    ],
    output_dict=True
)


print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Attack"
        ]
    )
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 — CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion matrix:")

print(cm)


# ------------------------------------------------------------
# EXTRACT CONFUSION MATRIX VALUES
# ------------------------------------------------------------

true_normal = int(
    cm[0][0]
)

false_attack = int(
    cm[0][1]
)

false_normal = int(
    cm[1][0]
)

true_attack = int(
    cm[1][1]
)


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 — FEATURE IMPORTANCE")
print("=" * 70)


importance = model.feature_importances_


feature_importance = []

for feature, value in zip(
    feature_columns,
    importance
):

    feature_importance.append({

        "feature": feature,

        "importance": float(value)

    })


feature_importance.sort(
    key=lambda item:
    item["importance"],
    reverse=True
)


print("\nTop 20 features:")

for item in feature_importance[:20]:

    print(
        f"{item['feature']:<35} "
        f"{item['importance']:.10f}"
    )


# ------------------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 10 — MODEL INFORMATION")
print("=" * 70)


model_information = {

    "algorithm":
        "Random Forest",

    "n_estimators":
        int(model.n_estimators),

    "random_state":
        int(model.random_state),

    "n_features":
        int(model.n_features_in_),

    "classes":
        [
            int(value)
            for value in model.classes_
        ],

    "predict_proba_supported":
        bool(predict_proba_supported)

}


print("\nAlgorithm:")
print(model_information["algorithm"])

print("\nNumber of trees:")
print(model_information["n_estimators"])

print("\nNumber of features:")
print(model_information["n_features"])

print("\nClasses:")
print(model_information["classes"])

print(
    "\npredict_proba supported:"
)

print(
    model_information[
        "predict_proba_supported"
    ]
)


# ------------------------------------------------------------
# CREATE RESULTS OBJECT
# ------------------------------------------------------------

results = {

    "project":
        "IoT Network Intrusion Detection "
        "and Security Monitoring System",

    "dataset":
        "N-BaIoT",

    "task":
        "Binary classification",

    "label_mapping": {

        "0":
            "Normal",

        "1":
            "Attack"

    },

    "dataset_split": {

        "test_samples":
            int(len(y_test)),

        "test_normal":
            int((y_test == 0).sum()),

        "test_attack":
            int((y_test == 1).sum())

    },

    "model":
        model_information,

    "metrics": {

        "accuracy":
            float(accuracy),

        "precision_attack":
            float(precision),

        "recall_attack":
            float(recall),

        "f1_score_attack":
            float(f1)

    },

    "classification_report":
        report,

    "confusion_matrix": {

        "matrix":
            cm.tolist(),

        "true_normal":
            true_normal,

        "false_attack":
            false_attack,

        "false_normal":
            false_normal,

        "true_attack":
            true_attack

    },

    "feature_importance":
        feature_importance

}


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 11 — SAVING RESULTS")
print("=" * 70)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\nResults saved to:")

print(RESULTS_FILE)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


checks = {

    "115 model features":
        len(feature_columns) == 115,

    "Test samples = 17,983":
        len(y_test) == 17983,

    "Accuracy available":
        isinstance(
            accuracy,
            float
        ),

    "Precision available":
        isinstance(
            precision,
            float
        ),

    "Recall available":
        isinstance(
            recall,
            float
        ),

    "F1-score available":
        isinstance(
            f1,
            float
        ),

    "Confusion matrix available":
        cm.shape == (2, 2),

    "Feature importance available":
        len(feature_importance) == 115,

    "Results file exists":
        os.path.exists(RESULTS_FILE)

}


all_passed = True

print()

for check_name, passed in checks.items():

    if passed:

        print(
            f"✓ {check_name}"
        )

    else:

        print(
            f"✗ {check_name}"
        )

        all_passed = False


print("\n" + "=" * 70)


if all_passed:

    print("SUCCESS!")

    print(
        "Evaluation results created successfully."
    )

else:

    print("WARNING!")

    print(
        "One or more evaluation checks failed."
    )


print("=" * 70)