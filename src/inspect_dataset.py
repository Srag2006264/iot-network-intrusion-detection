from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "N-BaIoT-Extracted"
)

MIRAI_PATH = (
    DATASET_PATH
    / "Danmini_Doorbell"
    / "mirai_extracted"
    / "mirai_attacks"
)


# =========================================================
# START
# =========================================================

print("=" * 70)
print("N-BaIoT MIRAI ATTACK SCHEMA VERIFICATION")
print("=" * 70)


# =========================================================
# CHECK MIRAI FOLDER
# =========================================================

if not MIRAI_PATH.exists():

    print("\nERROR: Mirai extraction folder not found.")
    print("\nExpected path:")
    print(MIRAI_PATH)

    raise SystemExit


print("\nMirai folder:")
print(MIRAI_PATH)


# =========================================================
# FIND CSV FILES
# =========================================================

csv_files = sorted(MIRAI_PATH.glob("*.csv"))


print("\nCSV files found:", len(csv_files))

for file in csv_files:
    print(" -", file.name)


# =========================================================
# CHECK WHETHER FILES EXIST
# =========================================================

if not csv_files:

    print("\nERROR: No CSV files found.")

    raise SystemExit


# =========================================================
# USE FIRST CSV AS REFERENCE
# =========================================================

reference_file = csv_files[0]

reference_columns = pd.read_csv(
    reference_file,
    nrows=0
).columns.tolist()


print("\n" + "=" * 70)
print("REFERENCE SCHEMA")
print("=" * 70)

print("\nReference file:")
print(reference_file.name)

print("\nNumber of columns:")
print(len(reference_columns))


# =========================================================
# COMPARE ALL MIRAI FILES
# =========================================================

print("\n" + "=" * 70)
print("COMPARING MIRAI FILES")
print("=" * 70)


all_match = True


for file in csv_files:

    # IMPORTANT:
    # nrows=0 means we read only the header.
    # The huge CSV data is NOT loaded into memory.

    columns = pd.read_csv(
        file,
        nrows=0
    ).columns.tolist()


    same_schema = (
        columns == reference_columns
    )


    print(f"\nFile: {file.name}")
    print(f"Columns: {len(columns)}")
    print(f"Exact same order: {same_schema}")


    # -----------------------------------------------------
    # SHOW DIFFERENCES IF THERE IS A MISMATCH
    # -----------------------------------------------------

    if not same_schema:

        all_match = False

        print("\nWARNING: Schema mismatch!")

        max_length = max(
            len(reference_columns),
            len(columns)
        )


        for i in range(max_length):

            if i < len(reference_columns):
                expected = reference_columns[i]
            else:
                expected = "<missing>"


            if i < len(columns):
                actual = columns[i]
            else:
                actual = "<missing>"


            if expected != actual:

                print(
                    f"Position {i}: "
                    f"expected '{expected}', "
                    f"found '{actual}'"
                )


# =========================================================
# SHOW COMPLETE FEATURE LIST
# =========================================================

print("\n" + "=" * 70)
print("REFERENCE FEATURE LIST")
print("=" * 70)


for i, feature in enumerate(reference_columns):

    print(f"{i}: {feature}")


# =========================================================
# FINAL RESULT
# =========================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)


if all_match:

    print("\nSUCCESS!")

    print(
        "All Mirai attack CSV files have "
        "the exact same feature schema and order."
    )

else:

    print("\nWARNING!")

    print(
        "At least one Mirai file has "
        "a different feature schema."
    )


print("\n" + "=" * 70)