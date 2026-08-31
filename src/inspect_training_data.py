import os
import pandas as pd


# ============================================================
# N-BaIoT TRAINING DATA QUALITY INSPECTION
# ============================================================

print("=" * 70)
print("N-BaIoT TRAINING DATA QUALITY INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATH
# ------------------------------------------------------------

TRAINING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "training_dataset.csv"
)

print("\nTraining dataset:")
print(TRAINING_FILE)


# ------------------------------------------------------------
# 2. CHECK FILE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 — FILE CHECK")
print("=" * 70)

if not os.path.exists(TRAINING_FILE):
    print("\nERROR: Training dataset was not found.")
    print("Expected location:")
    print(TRAINING_FILE)
    raise FileNotFoundError(TRAINING_FILE)

print("\nSUCCESS: Training dataset found.")

file_size_mb = os.path.getsize(TRAINING_FILE) / (1024 * 1024)

print(f"File size: {file_size_mb:.2f} MB")


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 — LOADING DATA")
print("=" * 70)

df = pd.read_csv(TRAINING_FILE)

print("\nDataset loaded successfully.")


# ------------------------------------------------------------
# 4. BASIC SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 — DATASET SHAPE")
print("=" * 70)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# 5. COLUMN INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 — COLUMN INFORMATION")
print("=" * 70)

print("\nFirst 10 columns:")

for i, column in enumerate(df.columns[:10]):
    print(f"{i}: {column}")

print("\nLast 10 columns:")

start = max(0, len(df.columns) - 10)

for i in range(start, len(df.columns)):
    print(f"{i}: {df.columns[i]}")


# ------------------------------------------------------------
# 6. EXPECTED COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 — EXPECTED COLUMNS")
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

print(f"\nModel feature columns: {len(feature_columns)}")

print(f"Metadata columns: {len(metadata_columns)}")

print("\nMetadata columns found:")

for column in metadata_columns:
    if column in df.columns:
        print(f"  ✓ {column}")
    else:
        print(f"  ✗ {column}")


# ------------------------------------------------------------
# 7. FEATURE COUNT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 — FEATURE COUNT CHECK")
print("=" * 70)

if len(feature_columns) == 115:
    print("\nSUCCESS: Exactly 115 model features found.")
else:
    print("\nERROR: Expected 115 model features.")
    print(f"Found: {len(feature_columns)}")


# ------------------------------------------------------------
# 8. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 — MISSING VALUE CHECK")
print("=" * 70)

total_missing = df.isnull().sum().sum()

print(f"\nTotal missing values: {total_missing:,}")

if total_missing == 0:
    print("SUCCESS: No missing values found.")
else:
    print("WARNING: Missing values found.")

    missing_columns = df.isnull().sum()

    print("\nColumns containing missing values:")

    print(
        missing_columns[missing_columns > 0]
    )


# ------------------------------------------------------------
# 9. LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 — LABEL DISTRIBUTION")
print("=" * 70)

print("\nNumeric labels:")

print(df["label"].value_counts().sort_index())


print("\nLabel names:")

print(df["label_name"].value_counts())


# ------------------------------------------------------------
# 10. ATTACK FAMILY DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 — ATTACK FAMILY DISTRIBUTION")
print("=" * 70)

print("\nAttack family distribution:")

print(df["attack_family"].value_counts())


# ------------------------------------------------------------
# 11. UNIQUE LABEL VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 10 — UNIQUE VALUES")
print("=" * 70)

print("\nUnique label values:")

print(sorted(df["label"].unique()))

print("\nUnique label names:")

print(sorted(df["label_name"].unique()))

print("\nUnique attack families:")

print(sorted(df["attack_family"].unique()))


# ------------------------------------------------------------
# 12. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 11 — DATA TYPE CHECK")
print("=" * 70)

feature_dtypes = df[feature_columns].dtypes

non_numeric_features = feature_dtypes[
    ~feature_dtypes.apply(
        lambda dtype: pd.api.types.is_numeric_dtype(dtype)
    )
]

print(f"\nTotal model features: {len(feature_columns)}")

print(
    f"Non-numeric model features: "
    f"{len(non_numeric_features)}"
)

if len(non_numeric_features) == 0:
    print("SUCCESS: All 115 model features are numeric.")
else:
    print("\nWARNING: Non-numeric features found:")

    print(non_numeric_features)


# ------------------------------------------------------------
# 13. DUPLICATE ROW CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 12 — DUPLICATE CHECK")
print("=" * 70)

duplicate_rows = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicate_rows:,}")


# ------------------------------------------------------------
# 14. FEATURE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 13 — FEATURE VALUE CHECK")
print("=" * 70)

feature_data = df[feature_columns]

print("\nChecking for infinite values...")

infinite_values = feature_data.isin(
    [float("inf"), float("-inf")]
).sum().sum()

print(f"Infinite values: {infinite_values:,}")

print("\nFeature statistics calculated successfully.")

print("\nFirst 5 feature statistics:")

print(
    feature_data.iloc[:, :5].describe().T
)


# ------------------------------------------------------------
# 15. SAMPLE DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 14 — SAMPLE DATA")
print("=" * 70)

print("\nFirst 5 rows:")

print(df.head())


# ------------------------------------------------------------
# 16. FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

checks = {
    "Rows = 99,096": len(df) == 99096,
    "115 model features": len(feature_columns) == 115,
    "No missing values": total_missing == 0,
    "All features numeric": len(non_numeric_features) == 0,
    "No infinite values": infinite_values == 0,
    "Label values are 0 and 1": sorted(df["label"].unique()) == [0, 1],
    "Normal and Attack labels exist": set(df["label_name"]) == {
        "Normal",
        "Attack"
    },
    "Attack families correct": set(df["attack_family"]) == {
        "Normal",
        "Gafgyt",
        "Mirai"
    }
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
    print("Training dataset passed all quality checks.")
else:
    print("WARNING!")
    print("One or more quality checks failed.")

print("=" * 70)