import os
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# N-BaIoT RANDOM FOREST TRAINING
# ============================================================

print("=" * 70)
print("N-BaIoT RANDOM FOREST TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "random_forest.joblib"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_columns.txt"
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

RANDOM_STATE = 42

N_ESTIMATORS = 200

N_JOBS = -1


print("\nRandom state:")
print(RANDOM_STATE)

print("\nNumber of trees:")
print(N_ESTIMATORS)

print("\nParallel jobs:")
print(N_JOBS)


# ------------------------------------------------------------
# CHECK INPUT FILES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — CHECKING INPUT FILES")
print("=" * 70)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Testing file not found:\n{TEST_FILE}"
    )

print("\n✓ Training dataset found.")
print("✓ Testing dataset found.")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — LOADING TRAINING AND TESTING DATA")
print("=" * 70)

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_FILE)

print("Training data loaded.")

print("\nLoading testing data...")

test_df = pd.read_csv(TEST_FILE)

print("Testing data loaded.")


print("\nTraining shape:")
print(train_df.shape)

print("\nTesting shape:")
print(test_df.shape)


# ------------------------------------------------------------
# IDENTIFY FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — IDENTIFYING MODEL FEATURES")
print("=" * 70)

feature_columns = [
    column
    for column in train_df.columns
    if column != "label"
]

print("\nModel features:")
print(len(feature_columns))

if len(feature_columns) != 115:
    raise ValueError(
        f"Expected 115 features, found {len(feature_columns)}"
    )

print("SUCCESS: Exactly 115 model features found.")


# ------------------------------------------------------------
# VERIFY TRAIN / TEST SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — VERIFYING FEATURE SCHEMA")
print("=" * 70)

test_features = [
    column
    for column in test_df.columns
    if column != "label"
]

if feature_columns != test_features:
    raise ValueError(
        "Training and testing feature schemas do not match."
    )

print("SUCCESS: Training and testing feature schemas match.")

print("SUCCESS: Feature order preserved.")


# ------------------------------------------------------------
# CREATE X AND Y
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — PREPARING X AND y")
print("=" * 70)

X_train = train_df[feature_columns]

y_train = train_df["label"]

X_test = test_df[feature_columns]

y_test = test_df["label"]


print("\nX_train shape:")
print(X_train.shape)

print("\ny_train shape:")
print(y_train.shape)

print("\nX_test shape:")
print(X_test.shape)

print("\ny_test shape:")
print(y_test.shape)


# ------------------------------------------------------------
# LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — LABEL DISTRIBUTION")
print("=" * 70)

print("\nTraining labels:")

print(
    y_train.value_counts().sort_index()
)

print("\nTesting labels:")

print(
    y_test.value_counts().sort_index()
)


# ------------------------------------------------------------
# CREATE RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — CREATING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

print("\nRandom Forest created successfully.")


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 — TRAINING RANDOM FOREST")
print("=" * 70)

print("\nTraining started...")

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("\nTraining completed.")

print(
    f"Training time: {training_time:.2f} seconds"
)


# ------------------------------------------------------------
# MAKE PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 — TESTING MODEL")
print("=" * 70)

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Predictions generated successfully.")


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 10 — MODEL ACCURACY")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 11 — CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion matrix:")

print(cm)

print("\nMatrix format:")

print(
    "[[True Normal,  False Attack]"
)

print(
    " [False Normal, True Attack]]"
)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 12 — CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal",
        "Attack"
    ]
)

print()

print(report)


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 13 — FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 important features:")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# CREATE MODEL DIRECTORY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 14 — SAVING MODEL")
print("=" * 70)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SAVE RANDOM FOREST
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print("\nRandom Forest saved to:")

print(MODEL_FILE)


# ------------------------------------------------------------
# SAVE FEATURE SCHEMA
# ------------------------------------------------------------

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as file:

    for feature in feature_columns:
        file.write(feature + "\n")


print("\nFeature schema saved to:")

print(FEATURE_FILE)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print()

checks = {
    "115 model features":
        len(feature_columns) == 115,

    "Training rows correct":
        len(X_train) == 71928,

    "Testing rows correct":
        len(X_test) == 17983,

    "Both classes in training":
        set(y_train.unique()) == {0, 1},

    "Both classes in testing":
        set(y_test.unique()) == {0, 1},

    "Model trained":
        hasattr(model, "estimators_"),

    "Model has correct tree count":
        len(model.estimators_) == N_ESTIMATORS,

    "Model file exists":
        os.path.exists(MODEL_FILE),

    "Feature file exists":
        os.path.exists(FEATURE_FILE)
}


all_passed = True

for check_name, passed in checks.items():

    if passed:
        print(f"✓ {check_name}")

    else:
        print(f"✗ {check_name}")
        all_passed = False


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)

if all_passed:

    print("SUCCESS!")
    print("Random Forest training completed successfully.")

else:

    print("WARNING!")
    print("One or more validation checks failed.")

print("=" * 70)