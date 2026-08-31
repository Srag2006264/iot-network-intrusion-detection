import os
import json
import joblib
import pandas as pd


# ============================================================
# RANDOM FOREST MODEL INTEGRATION TEST
# ============================================================

print("=" * 70)
print("RANDOM FOREST MODEL INTEGRATION TEST")
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

SCHEMA_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "schema",
    "feature_schema.json"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test.csv"
)


# ------------------------------------------------------------
# STEP 1 — CHECK FILES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — CHECKING FILES")
print("=" * 70)


if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )


if not os.path.exists(SCHEMA_FILE):

    raise FileNotFoundError(
        f"Schema not found:\n{SCHEMA_FILE}"
    )


if not os.path.exists(TEST_FILE):

    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_FILE}"
    )


print("\n✓ Model found.")
print("✓ Feature schema found.")
print("✓ Test dataset found.")


# ------------------------------------------------------------
# STEP 2 — LOAD MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — LOADING MODEL")
print("=" * 70)


model = joblib.load(
    MODEL_FILE
)


print("\nRandom Forest loaded successfully.")


# ------------------------------------------------------------
# STEP 3 — LOAD SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — LOADING FEATURE SCHEMA")
print("=" * 70)


with open(
    SCHEMA_FILE,
    "r",
    encoding="utf-8"
) as file:

    schema = json.load(
        file
    )


feature_names = schema[
    "feature_names"
]


expected_feature_count = schema[
    "number_of_features"
]


label_mapping = schema[
    "label_mapping"
]


print(
    f"\nSchema features: "
    f"{len(feature_names)}"
)

print(
    f"Expected features: "
    f"{expected_feature_count}"
)

print("\nLabel mapping:")

print(
    f"0 = {label_mapping['0']}"
)

print(
    f"1 = {label_mapping['1']}"
)


# ------------------------------------------------------------
# STEP 4 — VERIFY SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — VERIFYING SCHEMA")
print("=" * 70)


if len(feature_names) != 115:

    raise ValueError(
        "Schema does not contain exactly 115 features."
    )


if expected_feature_count != 115:

    raise ValueError(
        "Schema expected feature count is not 115."
    )


model_features = (
    model.feature_names_in_.tolist()
)


if feature_names != model_features:

    raise ValueError(
        "Schema feature order does not match "
        "the trained model."
    )


print(
    "\n✓ Schema contains 115 features."
)

print(
    "✓ Schema feature order matches model."
)


# ------------------------------------------------------------
# STEP 5 — LOAD ONE TEST SAMPLE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — LOADING TEST SAMPLE")
print("=" * 70)


test_df = pd.read_csv(
    TEST_FILE
)


# Select exactly one row

sample = test_df[
    feature_names
].iloc[[0]]


print(
    "\nSample shape:"
)

print(
    sample.shape
)


if sample.shape != (1, 115):

    raise ValueError(
        "Sample does not contain exactly "
        "1 row and 115 features."
    )


print(
    "\n✓ Sample contains exactly 115 features."
)


# ------------------------------------------------------------
# STEP 6 — VERIFY SAMPLE ORDER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — VERIFYING SAMPLE FEATURE ORDER")
print("=" * 70)


if sample.columns.tolist() != feature_names:

    raise ValueError(
        "Sample feature order does not match schema."
    )


print(
    "\n✓ Sample feature order matches schema."
)


# ------------------------------------------------------------
# STEP 7 — PREDICT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — RUNNING MODEL PREDICTION")
print("=" * 70)


prediction = model.predict(
    sample
)


print(
    "\nRaw prediction:"
)

print(
    prediction
)


predicted_label = int(
    prediction[0]
)


predicted_name = label_mapping[
    str(predicted_label)
]


print(
    "\nPredicted label:"
)

print(
    predicted_label
)


print(
    "\nPredicted class:"
)

print(
    predicted_name
)


# ------------------------------------------------------------
# STEP 8 — PREDICT PROBABILITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 — CALCULATING MODEL CONFIDENCE")
print("=" * 70)


if not hasattr(
    model,
    "predict_proba"
):

    raise ValueError(
        "Random Forest does not support predict_proba()."
    )


probabilities = model.predict_proba(
    sample
)


normal_probability = (
    float(probabilities[0][0])
)


attack_probability = (
    float(probabilities[0][1])
)


print(
    f"\nNormal probability: "
    f"{normal_probability:.6f}"
)

print(
    f"Attack probability: "
    f"{attack_probability:.6f}"
)


# ------------------------------------------------------------
# STEP 9 — FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL RESULT")
print("=" * 70)


print(
    f"\nPrediction: "
    f"{predicted_name}"
)


if predicted_label == 1:

    print(
        f"Attack confidence: "
        f"{attack_probability * 100:.2f}%"
    )

else:

    print(
        f"Normal confidence: "
        f"{normal_probability * 100:.2f}%"
    )


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


checks = {

    "Model loaded":
        model is not None,

    "Schema loaded":
        schema is not None,

    "115 features":
        len(feature_names) == 115,

    "Model has 115 features":
        model.n_features_in_ == 115,

    "Schema matches model":
        feature_names == model_features,

    "Sample shape is 1 x 115":
        sample.shape == (1, 115),

    "Prediction generated":
        len(prediction) == 1,

    "Prediction is valid":
        predicted_label in [0, 1],

    "predict_proba supported":
        hasattr(
            model,
            "predict_proba"
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
        "Random Forest integration test passed."
    )

else:

    print("WARNING!")

    print(
        "One or more integration checks failed."
    )


print("=" * 70)