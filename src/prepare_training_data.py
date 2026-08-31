from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

TARGET_NORMAL = 49_548
TARGET_ATTACK = 49_548

# Based on the Danmini Doorbell inventory:
# Gafgyt = 316,650
# Mirai  = 652,100
#
# We preserve approximately the same family proportion.

TARGET_GAFGYT = 16_195
TARGET_MIRAI = 33_353

CHUNK_SIZE = 100_000


# =========================================================
# DATASET PATHS
# =========================================================

DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "N-BaIoT-Extracted"
)

DANMINI_PATH = (
    DATASET_PATH
    / "Danmini_Doorbell"
)

BENIGN_FILE = (
    DANMINI_PATH
    / "benign_traffic.csv"
)

GAFGYT_PATH = (
    DANMINI_PATH
    / "gafgyt_extracted"
    / "gafgyt_attacks"
)

MIRAI_PATH = (
    DANMINI_PATH
    / "mirai_extracted"
    / "mirai_attacks"
)


# =========================================================
# OUTPUT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "training_dataset.csv"
)


# =========================================================
# RESERVOIR SAMPLING FUNCTION
# =========================================================

def reservoir_sample_csv_files(
    files,
    target_size,
    family_name
):
    """
    Read CSV files in chunks and keep a reproducible
    random sample without loading all rows into RAM.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    reservoir = None

    total_seen = 0

    for file in files:

        print("\nReading:")
        print(file)

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE
        ):

            chunk_size = len(chunk)

            if chunk_size == 0:
                continue

            # -------------------------------------------------
            # Add attack-family metadata
            # -------------------------------------------------

            chunk["attack_family"] = family_name

            # -------------------------------------------------
            # Generate random scores
            # -------------------------------------------------

            scores = rng.random(
                chunk_size
            )

            chunk["_random_score"] = scores

            total_seen += chunk_size

            # -------------------------------------------------
            # First chunk / initial reservoir
            # -------------------------------------------------

            if reservoir is None:

                if len(chunk) <= target_size:

                    reservoir = chunk.copy()

                else:

                    reservoir = (
                        chunk
                        .nsmallest(
                            target_size,
                            "_random_score"
                        )
                        .copy()
                    )

            else:

                # -------------------------------------------------
                # Combine current reservoir + new chunk
                # -------------------------------------------------

                combined = pd.concat(
                    [
                        reservoir,
                        chunk
                    ],
                    ignore_index=True
                )

                # -------------------------------------------------
                # Keep only target_size random rows
                # -------------------------------------------------

                reservoir = (
                    combined
                    .nsmallest(
                        target_size,
                        "_random_score"
                    )
                    .copy()
                )

            print(
                f"Rows processed: "
                f"{total_seen:,}",
                end="\r"
            )

    print()

    if reservoir is None:

        raise ValueError(
            f"No data found for {family_name}."
        )

    # Remove temporary random column

    reservoir = reservoir.drop(
        columns=["_random_score"]
    )

    print(
        f"{family_name} samples selected: "
        f"{len(reservoir):,}"
    )

    return reservoir


# =========================================================
# START
# =========================================================

print("=" * 70)
print("N-BaIoT MEMORY-SAFE TRAINING DATA PREPARATION")
print("=" * 70)

print("\nRandom state:", RANDOM_STATE)

print("Chunk size:", CHUNK_SIZE)

print("\nTarget Normal samples:", TARGET_NORMAL)

print("Target Attack samples:", TARGET_ATTACK)

print("Target Gafgyt samples:", TARGET_GAFGYT)

print("Target Mirai samples:", TARGET_MIRAI)


# =========================================================
# STEP 1 — LOAD NORMAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 1 — NORMAL TRAFFIC")
print("=" * 70)

print("\nLoading:")
print(BENIGN_FILE)


normal_df = pd.read_csv(
    BENIGN_FILE
)


print("\nOriginal rows:")
print(len(normal_df))

print("Original columns:")
print(len(normal_df.columns))


# =========================================================
# VERIFY NORMAL DATA
# =========================================================

if len(normal_df) < TARGET_NORMAL:

    raise ValueError(
        "Not enough Normal samples."
    )


# =========================================================
# SAMPLE NORMAL
# =========================================================

normal_df = normal_df.sample(
    n=TARGET_NORMAL,
    random_state=RANDOM_STATE
).copy()


# =========================================================
# ADD LABELS
# =========================================================

normal_df["label"] = 0

normal_df["label_name"] = "Normal"

normal_df["attack_family"] = "Normal"


print("\nSelected Normal samples:")
print(len(normal_df))


# =========================================================
# STEP 2 — FIND ATTACK FILES
# =========================================================

print("\n" + "=" * 70)
print("STEP 2 — ATTACK FILES")
print("=" * 70)


gafgyt_files = sorted(
    GAFGYT_PATH.glob("*.csv")
)

mirai_files = sorted(
    MIRAI_PATH.glob("*.csv")
)


print("\nGafgyt files:")

for file in gafgyt_files:
    print(" -", file.name)


print("\nMirai files:")

for file in mirai_files:
    print(" -", file.name)


# =========================================================
# STEP 3 — SAMPLE GAFGYT
# =========================================================

print("\n" + "=" * 70)
print("STEP 3 — SAMPLING GAFGYT")
print("=" * 70)


gafgyt_df = reservoir_sample_csv_files(
    gafgyt_files,
    TARGET_GAFGYT,
    "Gafgyt"
)


# =========================================================
# ADD GAFGYT LABEL
# =========================================================

gafgyt_df["label"] = 1

gafgyt_df["label_name"] = "Attack"


# =========================================================
# STEP 4 — SAMPLE MIRAI
# =========================================================

print("\n" + "=" * 70)
print("STEP 4 — SAMPLING MIRAI")
print("=" * 70)


mirai_df = reservoir_sample_csv_files(
    mirai_files,
    TARGET_MIRAI,
    "Mirai"
)


# =========================================================
# ADD MIRAI LABEL
# =========================================================

mirai_df["label"] = 1

mirai_df["label_name"] = "Attack"


# =========================================================
# STEP 5 — COMBINE ATTACK DATA
# =========================================================

print("\n" + "=" * 70)
print("STEP 5 — COMBINING ATTACK DATA")
print("=" * 70)


attack_df = pd.concat(
    [
        gafgyt_df,
        mirai_df
    ],
    ignore_index=True
)


print("\nTotal Attack samples:")
print(len(attack_df))


# =========================================================
# VERIFY ATTACK COUNT
# =========================================================

if len(attack_df) != TARGET_ATTACK:

    raise ValueError(
        "Attack sample count is incorrect."
    )


# =========================================================
# STEP 6 — COMBINE NORMAL + ATTACK
# =========================================================

print("\n" + "=" * 70)
print("STEP 6 — COMBINING NORMAL AND ATTACK")
print("=" * 70)


training_df = pd.concat(
    [
        normal_df,
        attack_df
    ],
    ignore_index=True
)


# =========================================================
# SHUFFLE
# =========================================================

training_df = training_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


# =========================================================
# STEP 7 — BASIC DATA CLEANING
# =========================================================

print("\n" + "=" * 70)
print("STEP 7 — DATA CLEANING")
print("=" * 70)


print("\nRows before cleaning:")
print(len(training_df))


# ---------------------------------------------------------
# Replace infinite values
# ---------------------------------------------------------

training_df = training_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ---------------------------------------------------------
# Count missing values
# ---------------------------------------------------------

missing_before = (
    training_df.isna()
    .sum()
    .sum()
)


print("\nMissing values:")
print(missing_before)


# ---------------------------------------------------------
# Remove rows containing NaN
# ---------------------------------------------------------

if missing_before > 0:

    training_df = training_df.dropna()


print("\nRows after cleaning:")
print(len(training_df))


# =========================================================
# STEP 8 — CHECK FEATURE COUNT
# =========================================================

print("\n" + "=" * 70)
print("STEP 8 — FEATURE SCHEMA CHECK")
print("=" * 70)


metadata_columns = [
    "label",
    "label_name",
    "attack_family"
]


feature_columns = [
    column
    for column in training_df.columns
    if column not in metadata_columns
]


print("\nNumber of model features:")
print(len(feature_columns))


if len(feature_columns) != 115:

    raise ValueError(
        "ERROR: Expected exactly 115 model features."
    )


print("\nSUCCESS: 115 model features confirmed.")


# =========================================================
# SHOW COMPLETE FEATURE ORDER
# =========================================================

print("\nFeature order:")

for i, feature in enumerate(
    feature_columns
):

    print(
        f"{i}: {feature}"
    )


# =========================================================
# STEP 9 — CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("STEP 9 — CLASS DISTRIBUTION")
print("=" * 70)


print(
    training_df["label_name"]
    .value_counts()
)


print("\nAttack family distribution:")

print(
    training_df["attack_family"]
    .value_counts()
)


# =========================================================
# STEP 10 — SAVE DATASET
# =========================================================

print("\n" + "=" * 70)
print("STEP 10 — SAVING DATASET")
print("=" * 70)


training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nTraining dataset saved to:")

print(OUTPUT_FILE)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("TRAINING DATASET PREPARATION COMPLETE")
print("=" * 70)


print("\nFinal rows:")
print(len(training_df))

print("\nModel features:")
print(len(feature_columns))

print("\nLabel mapping:")
print("0 = Normal")
print("1 = Attack")

print("\nAttack families:")
print("Gafgyt")
print("Mirai")

print("\nOutput file:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)