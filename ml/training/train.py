import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# REPRODUCIBLE N-BaIoT RANDOM FOREST TRAINING
# ============================================================

print("=" * 70)
print("REPRODUCIBLE N-BaIoT RANDOM FOREST TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ------------------------------------------------------------
# INPUT / OUTPUT PATHS
# ------------------------------------------------------------

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "model"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "random_forest.joblib"
)

SCHEMA_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "schema"
)

SCHEMA_FILE = os.path.join(
    SCHEMA_DIR,
    "feature_schema.json"
)


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42

N_ESTIMATORS = 200

N_JOBS = -1


# ------------------------------------------------------------
# DISPLAY CONFIGURATION
# ------------------------------------------------------------

print("\nDataset:")
print("N-BaIoT")

print("\nAlgorithm:")
print("Random Forest")

print("\nNumber of trees:")
print(N_ESTIMATORS)

print("\nRandom state:")
print(RANDOM_STATE)

print("\nScaling:")
print("None")

print("\nFeature selection:")
print("None")


# ------------------------------------------------------------
# CHECK TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — CHECKING TRAINING DATA")
print("=" * 70)

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_FILE}"
    )

print("\n✓ Training dataset found.")

print(TRAIN_FILE)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — LOADING TRAINING DATA")
print("=" * 70)

df = pd.read_csv(
    TRAIN_FILE
)

print("\nTraining data loaded.")

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ------------------------------------------------------------
# IDENTIFY FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — IDENTIFYING FEATURES")
print("=" * 70)

feature_columns = [
    column
    for column in df.columns
    if column != "label"
]


print(
    f"\nNumber of features: "
    f"{len(feature_columns)}"
)


if len(feature_columns) != 115:

    raise ValueError(
        f"Expected 115 features, "
        f"found {len(feature_columns)}."
    )


print(
    "SUCCESS: Exactly 115 features found."
)


# ------------------------------------------------------------
# PREPARE X AND y
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — PREPARING TRAINING DATA")
print("=" * 70)


X_train = df[
    feature_columns
]

y_train = df[
    "label"
]


print("\nX_train:")
print(X_train.shape)

print("\ny_train:")
print(y_train.shape)


# ------------------------------------------------------------
# VERIFY LABELS
# ------------------------------------------------------------

unique_labels = sorted(
    y_train.unique().tolist()
)


print("\nLabels:")
print(unique_labels)


if unique_labels != [0, 1]:

    raise ValueError(
        "Expected labels 0 and 1."
    )


print("\nLabel mapping:")
print("0 = Normal")
print("1 = Attack")


# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — CREATING RANDOM FOREST")
print("=" * 70)


model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS
)


print("\nRandom Forest created successfully.")


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — TRAINING MODEL")
print("=" * 70)


print("\nTraining started...")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed.")


# ------------------------------------------------------------
# VERIFY MODEL SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — VERIFYING MODEL SCHEMA")
print("=" * 70)


model_features = (
    model.feature_names_in_.tolist()
)


if model_features != feature_columns:

    raise ValueError(
        "Model feature order does not "
        "match training feature order."
    )


print(
    "\nSUCCESS: Model feature names "
    "and order preserved."
)


print(
    f"Model features: "
    f"{model.n_features_in_}"
)


# ------------------------------------------------------------
# CREATE DIRECTORIES
# ------------------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    SCHEMA_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 — SAVING MODEL")
print("=" * 70)


joblib.dump(
    model,
    MODEL_FILE
)


print("\nModel saved to:")

print(MODEL_FILE)


# ------------------------------------------------------------
# CREATE SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 — SAVING FEATURE SCHEMA")
print("=" * 70)


schema = {

    "dataset":
        "N-BaIoT",

    "algorithm":
        "Random Forest",

    "number_of_features":
        int(model.n_features_in_),

    "feature_names":
        model_features,

    "feature_order":
        model_features,

    "label_mapping": {

        "0":
            "Normal",

        "1":
            "Attack"

    },

    "preprocessing": {

        "missing_values_removed":
            True,

        "infinite_values_removed":
            True,

        "exact_duplicate_rows_removed":
            True,

        "scaling":
            False,

        "normalization":
            False,

        "feature_selection":
            False

    },

    "model_parameters": {

        "n_estimators":
            int(model.n_estimators),

        "random_state":
            int(model.random_state),

        "n_jobs":
            int(model.n_jobs)

    },

    "predict_proba_supported":
        hasattr(
            model,
            "predict_proba"
        )

}


with open(
    SCHEMA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        schema,
        file,
        indent=4
    )


print("\nFeature schema saved to:")

print(SCHEMA_FILE)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


checks = {

    "115 features":
        model.n_features_in_ == 115,

    "Feature order preserved":
        model_features == feature_columns,

    "Classes are 0 and 1":
        model.classes_.tolist() == [0, 1],

    "200 trees":
        model.n_estimators == 200,

    "predict_proba supported":
        hasattr(
            model,
            "predict_proba"
        ),

    "Model file exists":
        os.path.exists(
            MODEL_FILE
        ),

    "Schema file exists":
        os.path.exists(
            SCHEMA_FILE
        )

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
        "Reproducible training package "
        "created successfully."
    )

else:

    print("WARNING!")

    print(
        "One or more validation checks failed."
    )


print("=" * 70)