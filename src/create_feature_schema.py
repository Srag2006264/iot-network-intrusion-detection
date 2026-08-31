import os
import json
import joblib


# ============================================================
# CREATE FINAL RANDOM FOREST FEATURE SCHEMA
# ============================================================

print("=" * 70)
print("CREATING FINAL RANDOM FOREST FEATURE SCHEMA")
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
# CHECK MODEL
# ------------------------------------------------------------

print("\nChecking trained model...")

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"\nRandom Forest model not found:\n{MODEL_FILE}"
    )

print("SUCCESS: Random Forest model found.")

print("\nModel location:")
print(MODEL_FILE)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — LOADING RANDOM FOREST")
print("=" * 70)

model = joblib.load(
    MODEL_FILE
)

print("\nRandom Forest loaded successfully.")


# ------------------------------------------------------------
# GET FEATURE NAMES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — EXTRACTING FEATURE SCHEMA")
print("=" * 70)

if not hasattr(
    model,
    "feature_names_in_"
):

    raise ValueError(
        "The trained model does not contain "
        "feature_names_in_."
    )


feature_names = (
    model.feature_names_in_.tolist()
)


number_of_features = (
    model.n_features_in_
)


print("\nNumber of features:")
print(number_of_features)


if number_of_features != 115:

    raise ValueError(
        f"Expected 115 features, "
        f"but model contains {number_of_features}."
    )


print(
    "\nSUCCESS: Model expects exactly "
    "115 features."
)


# ------------------------------------------------------------
# DISPLAY COMPLETE FEATURE ORDER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — EXACT FEATURE ORDER")
print("=" * 70)

for index, feature in enumerate(
    feature_names
):

    print(
        f"{index}: {feature}"
    )


# ------------------------------------------------------------
# MODEL CLASSES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — MODEL CLASS INFORMATION")
print("=" * 70)

model_classes = [
    int(value)
    for value in model.classes_
]

print("\nModel classes:")

print(model_classes)


# ------------------------------------------------------------
# LABEL MAPPING
# ------------------------------------------------------------

label_mapping = {
    "0": "Normal",
    "1": "Attack"
}


print("\nLabel mapping:")

print("0 = Normal")
print("1 = Attack")


# ------------------------------------------------------------
# PREPROCESSING INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — PREPROCESSING INFORMATION")
print("=" * 70)

preprocessing = {

    "missing_values_removed": True,

    "infinite_values_removed": True,

    "exact_duplicate_rows_removed": True,

    "scaling": False,

    "normalization": False,

    "feature_selection": False,

    "feature_selection_description":
        "No arbitrary feature reduction. "
        "All 115 N-BaIoT features were used."
}


print("\nPreprocessing:")

for key, value in preprocessing.items():

    print(
        f"{key}: {value}"
    )


# ------------------------------------------------------------
# PREDICT PROBA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — PREDICT_PROBA CHECK")
print("=" * 70)

predict_proba_supported = hasattr(
    model,
    "predict_proba"
)


print(
    "\npredict_proba supported:",
    predict_proba_supported
)


# ------------------------------------------------------------
# CREATE SCHEMA OBJECT
# ------------------------------------------------------------

schema = {

    "project":
        "IoT Network Intrusion Detection "
        "and Security Monitoring System",

    "dataset":
        "N-BaIoT",

    "model":
        "Random Forest",

    "model_file":
        "ml/model/random_forest.joblib",

    "number_of_features":
        number_of_features,

    "feature_names":
        feature_names,

    "feature_order":
        feature_names,

    "label_mapping":
        label_mapping,

    "model_classes":
        model_classes,

    "preprocessing":
        preprocessing,

    "prediction_probability_supported":
        predict_proba_supported

}


# ------------------------------------------------------------
# CREATE SCHEMA DIRECTORY
# ------------------------------------------------------------

os.makedirs(
    SCHEMA_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — SAVING FEATURE SCHEMA")
print("=" * 70)

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
        number_of_features == 115,

    "Feature names available":
        len(feature_names) == 115,

    "Feature order available":
        len(feature_names) == 115,

    "Model classes are 0 and 1":
        model_classes == [0, 1],

    "predict_proba supported":
        predict_proba_supported,

    "Schema file exists":
        os.path.exists(SCHEMA_FILE)

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
        "Final feature schema created successfully."
    )

else:

    print("WARNING!")
    print(
        "One or more schema checks failed."
    )

print("=" * 70)