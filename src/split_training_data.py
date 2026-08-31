import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# N-BaIoT TRAIN / TEST SPLIT
# ============================================================

print("=" * 70)
print("N-BaIoT TRAIN / TEST SPLIT")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "training_dataset_clean.csv"
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


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42


print("\nInput dataset:")
print(INPUT_FILE)

print("\nTest size:")
print(f"{TEST_SIZE * 100:.0f}%")

print("\nRandom state:")
print(RANDOM_STATE)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — LOADING CLEAN DATASET")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Clean dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# IDENTIFY FEATURES AND LABEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — IDENTIFYING FEATURES AND LABEL")
print("=" * 70)

metadata_columns = [
    "label",
    "label_name",
    "attack_family"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

print(f"\nModel features: {len(feature_columns)}")

if len(feature_columns) != 115:
    raise ValueError(
        f"Expected 115 model features, "
        f"found {len(feature_columns)}"
    )

print("SUCCESS: 115 model features confirmed.")

print("\nLabel column:")
print("label")

print("\nLabel mapping:")
print("0 = Normal")
print("1 = Attack")


# ------------------------------------------------------------
# PREPARE X AND y
# ------------------------------------------------------------

X = df[feature_columns]

y = df["label"]


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — STRATIFIED TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ------------------------------------------------------------
# SPLIT SIZES
# ------------------------------------------------------------

print("\nTraining samples:")
print(f"{len(X_train):,}")

print("\nTesting samples:")
print(f"{len(X_test):,}")

print("\nTotal:")
print(f"{len(X_train) + len(X_test):,}")


# ------------------------------------------------------------
# CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining labels:")

print(
    y_train
    .value_counts()
    .sort_index()
)


print("\nTraining percentages:")

print(
    (
        y_train
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


print("\n" + "=" * 70)
print("STEP 5 — TESTING CLASS DISTRIBUTION")
print("=" * 70)

print("\nTesting labels:")

print(
    y_test
    .value_counts()
    .sort_index()
)


print("\nTesting percentages:")

print(
    (
        y_test
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# ------------------------------------------------------------
# VERIFY SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — FEATURE SCHEMA VERIFICATION")
print("=" * 70)

if list(X_train.columns) != feature_columns:
    raise ValueError(
        "Training feature order changed!"
    )

if list(X_test.columns) != feature_columns:
    raise ValueError(
        "Testing feature order changed!"
    )

print(
    "\nSUCCESS: Training feature order preserved."
)

print(
    "SUCCESS: Testing feature order preserved."
)

print(
    f"Feature count: {len(feature_columns)}"
)


# ------------------------------------------------------------
# SAVE TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — SAVING TRAINING DATA")
print("=" * 70)

train_df = X_train.copy()

train_df["label"] = y_train.values

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

print("\nTraining dataset saved:")
print(TRAIN_FILE)


# ------------------------------------------------------------
# SAVE TEST DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 — SAVING TEST DATA")
print("=" * 70)

test_df = X_test.copy()

test_df["label"] = y_test.values

test_df.to_csv(
    TEST_FILE,
    index=False
)

print("\nTesting dataset saved:")
print(TEST_FILE)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

checks = {
    "Total rows preserved":
        len(X_train) + len(X_test) == len(df),

    "115 training features":
        X_train.shape[1] == 115,

    "115 testing features":
        X_test.shape[1] == 115,

    "Training schema preserved":
        list(X_train.columns) == feature_columns,

    "Testing schema preserved":
        list(X_test.columns) == feature_columns,

    "Training contains both classes":
        set(y_train.unique()) == {0, 1},

    "Testing contains both classes":
        set(y_test.unique()) == {0, 1}
}


print()

all_passed = True

for check_name, passed in checks.items():

    if passed:
        print(f"✓ {check_name}")
    else:
        print(f"✗ {check_name}")
        all_passed = False


print("\n" + "=" * 70)

if all_passed:
    print("SUCCESS!")
    print("Train/test split completed successfully.")
else:
    print("WARNING!")
    print("One or more validation checks failed.")

print("=" * 70)