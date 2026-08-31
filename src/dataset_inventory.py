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
# FUNCTION TO COUNT ROWS
# =========================================================

def count_rows(file_path):

    print(f"\nCounting rows in:")
    print(file_path)

    row_count = 0

    # Read the file in chunks so we do NOT load
    # the entire CSV into memory.

    for chunk in pd.read_csv(
        file_path,
        chunksize=100_000
    ):

        row_count += len(chunk)

        print(
            f"  Rows counted so far: {row_count:,}",
            end="\r"
        )

    print()

    return row_count


# =========================================================
# START
# =========================================================

print("=" * 70)
print("N-BaIoT DATASET INVENTORY")
print("=" * 70)


# =========================================================
# BENIGN
# =========================================================

print("\n")
print("=" * 70)
print("BENIGN TRAFFIC")
print("=" * 70)


benign_rows = count_rows(BENIGN_FILE)


print(
    f"Danmini_Doorbell benign rows: "
    f"{benign_rows:,}"
)


# =========================================================
# GAFGYT
# =========================================================

print("\n")
print("=" * 70)
print("GAFGYT ATTACKS")
print("=" * 70)


gafgyt_files = sorted(
    GAFGYT_PATH.glob("*.csv")
)


gafgyt_total = 0


for file in gafgyt_files:

    rows = count_rows(file)

    gafgyt_total += rows

    print(
        f"{file.name}: {rows:,} rows"
    )


print(
    f"\nTotal Gafgyt rows: "
    f"{gafgyt_total:,}"
)


# =========================================================
# MIRAI
# =========================================================

print("\n")
print("=" * 70)
print("MIRAI ATTACKS")
print("=" * 70)


mirai_files = sorted(
    MIRAI_PATH.glob("*.csv")
)


mirai_total = 0


for file in mirai_files:

    rows = count_rows(file)

    mirai_total += rows

    print(
        f"{file.name}: {rows:,} rows"
    )


print(
    f"\nTotal Mirai rows: "
    f"{mirai_total:,}"
)


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)


print(
    f"\nNormal samples: "
    f"{benign_rows:,}"
)

print(
    f"Attack samples: "
    f"{gafgyt_total + mirai_total:,}"
)

print(
    f"    Gafgyt: "
    f"{gafgyt_total:,}"
)

print(
    f"    Mirai: "
    f"{mirai_total:,}"
)

print(
    f"\nTotal samples: "
    f"{benign_rows + gafgyt_total + mirai_total:,}"
)


print("\n" + "=" * 70)
print("INVENTORY COMPLETE")
print("=" * 70)