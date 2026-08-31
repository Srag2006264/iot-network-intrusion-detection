import os
import pandas as pd


# ============================================================
# N-BaIoT TRAINING DATA DUPLICATE CLEANING
# ============================================================

print("=" * 70)
print("N-BaIoT TRAINING DATA DUPLICATE CLEANING")
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
    "training_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "training_dataset_clean.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nInput file:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Training dataset not found:\n{INPUT_FILE}"
    )

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — ORIGINAL DATASET")
print("=" * 70)

print(f"\nOriginal rows: {len(df):,}")
print(f"Original columns: {len(df.columns)}")


# ------------------------------------------------------------
# IDENTIFY COLUMNS
# ------------------------------------------------------------

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
        f"Expected 115 features but found "
        f"{len(feature_columns)}"
    )


# ------------------------------------------------------------
# DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_mask = df.duplicated(
    keep="first"
)

duplicate_count = duplicate_mask.sum()

print(f"\nExact duplicate rows: {duplicate_count:,}")

print(
    f"Percentage duplicated: "
    f"{(duplicate_count / len(df)) * 100:.2f}%"
)


# ------------------------------------------------------------
# CHECK DUPLICATES ACROSS FEATURES
# ------------------------------------------------------------

print("\nChecking feature-only duplicates...")

feature_duplicate_mask = df.duplicated(
    subset=feature_columns,
    keep=False
)

feature_duplicate_count = feature_duplicate_mask.sum()

print(
    f"Rows involved in feature duplicates: "
    f"{feature_duplicate_count:,}"
)


# ------------------------------------------------------------
# CHECK CONFLICTING LABELS
# ------------------------------------------------------------

print("\nChecking for identical feature vectors "
      "with different labels...")

feature_label_counts = (
    df.groupby(feature_columns, sort=False)["label"]
    .nunique()
)

conflicting_feature_vectors = (
    feature_label_counts > 1
).sum()

print(
    f"Feature vectors with conflicting labels: "
    f"{conflicting_feature_vectors:,}"
)


# ------------------------------------------------------------
# REMOVE EXACT DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — REMOVING EXACT DUPLICATES")
print("=" * 70)

df_clean = df.drop_duplicates(
    keep="first"
).reset_index(drop=True)

print(f"\nRows before cleaning: {len(df):,}")
print(f"Rows after cleaning:  {len(df_clean):,}")
print(f"Rows removed:        {len(df) - len(df_clean):,}")


# ------------------------------------------------------------
# VERIFY NO EXACT DUPLICATES REMAIN
# ------------------------------------------------------------

remaining_duplicates = df_clean.duplicated().sum()

print(
    f"\nRemaining exact duplicates: "
    f"{remaining_duplicates:,}"
)

if remaining_duplicates != 0:
    raise ValueError(
        "Duplicate rows still remain."
    )

print("SUCCESS: No exact duplicate rows remain.")


# ------------------------------------------------------------
# VERIFY FEATURE COUNT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — FEATURE SCHEMA VERIFICATION")
print("=" * 70)

clean_feature_columns = [
    column
    for column in df_clean.columns
    if column not in metadata_columns
]

print(
    f"\nModel features: "
    f"{len(clean_feature_columns)}"
)

if clean_feature_columns != feature_columns:
    raise ValueError(
        "Feature names or feature order changed!"
    )

print(
    "SUCCESS: Feature names and order preserved."
)


# ------------------------------------------------------------
# LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — LABEL DISTRIBUTION")
print("=" * 70)

print("\nNumeric labels:")

print(
    df_clean["label"]
    .value_counts()
    .sort_index()
)

print("\nLabel names:")

print(
    df_clean["label_name"]
    .value_counts()
)

print("\nAttack families:")

print(
    df_clean["attack_family"]
    .value_counts()
)


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — DATA QUALITY CHECK")
print("=" * 70)

missing_values = df_clean.isnull().sum().sum()

print(
    f"\nMissing values: "
    f"{missing_values:,}"
)

if missing_values != 0:
    raise ValueError(
        "Missing values detected after cleaning."
    )

print("SUCCESS: No missing values.")


# ------------------------------------------------------------
# INFINITE VALUES
# ------------------------------------------------------------

infinite_values = (
    df_clean[clean_feature_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print(
    f"Infinite values: "
    f"{infinite_values:,}"
)

if infinite_values != 0:
    raise ValueError(
        "Infinite values detected."
    )

print("SUCCESS: No infinite values.")


# ------------------------------------------------------------
# SAVE CLEAN DATASET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — SAVING CLEAN DATASET")
print("=" * 70)

df_clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nClean dataset saved to:")

print(OUTPUT_FILE)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"\nOriginal rows: "
    f"{len(df):,}"
)

print(
    f"Clean rows: "
    f"{len(df_clean):,}"
)

print(
    f"Duplicates removed: "
    f"{len(df) - len(df_clean):,}"
)

print(
    f"Model features: "
    f"{len(clean_feature_columns)}"
)

print(
    "\nLabel mapping:"
)

print("0 = Normal")
print("1 = Attack")

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)