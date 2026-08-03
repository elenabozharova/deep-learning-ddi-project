"""
Milestone 1 — Dataset Inspection
=================================
Loads ChChSe-Decagon_polypharmacy.csv.gz and prints structural information
needed to understand the task before any modeling decisions are made.

Run from the project root:
    python src/inspect_dataset.py

Output:
    - Printed summary to the terminal
    - data/processed/sample.csv  (500 reproducible rows)
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
SAMPLE_SIZE = 500

RAW_PATH = Path("data/raw/ChChSe-Decagon_polypharmacy.csv.gz")
SAMPLE_OUT = Path("data/processed/sample.csv")

# ---------------------------------------------------------------------------
# Helper: section header
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the compressed CSV.
    pandas reads .gz files directly — no manual decompression needed.
    Think of it like fetch() automatically following a redirect.
    """
    if not path.exists():
        sys.exit(
            f"\nERROR: File not found — {path}\n"
            "Place ChChSe-Decagon_polypharmacy.csv.gz inside data/raw/ and try again."
        )
    print(f"Loading: {path}")
    df = pd.read_csv(path, compression="gzip")
    print(f"Loaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Basic structure
# ---------------------------------------------------------------------------

def print_basic_info(df: pd.DataFrame) -> None:
    """Shape, column names, and dtypes — the first thing to check."""
    _header("BASIC STRUCTURE")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")
    print()
    print(f"{'Column name':<40}  dtype")
    print("-" * 55)
    for col in df.columns:
        print(f"  {col!r:<38}  {df[col].dtype}")


# ---------------------------------------------------------------------------
# Step 3 — Missing values
# ---------------------------------------------------------------------------

def print_missing_values(df: pd.DataFrame) -> None:
    """Missing values per column. Important before any modelling."""
    _header("MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
        return
    for col, count in missing.items():
        if count > 0:
            pct = 100 * count / len(df)
            print(f"  {col!r}: {count:,} missing  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Step 4 — First rows
# ---------------------------------------------------------------------------

def print_first_rows(df: pd.DataFrame, n: int = 10) -> None:
    """
    Print the first n rows so we can read actual values.
    Like console.log(data.slice(0, 10)) when debugging a JSON response.
    """
    _header(f"FIRST {n} ROWS")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.max_colwidth", 50,
    ):
        print(df.head(n).to_string(index=False))


# ---------------------------------------------------------------------------
# Step 5 — Unique value counts
# ---------------------------------------------------------------------------

def print_unique_counts(df: pd.DataFrame) -> None:
    """
    How many distinct values are in each column?
    This tells us whether side effects are a small fixed set or a large open vocabulary.
    """
    _header("UNIQUE VALUES PER COLUMN")
    for col in df.columns:
        n_unique = df[col].nunique()
        example = df[col].dropna().iloc[0] if len(df) > 0 else "—"
        print(f"  {col!r:<40}  {n_unique:>7,} unique   (first value: {example!r})")


# ---------------------------------------------------------------------------
# Step 6 — Drug-pair analysis
# ---------------------------------------------------------------------------

def print_drug_pair_analysis(df: pd.DataFrame) -> None:
    """
    Check whether the same drug pair appears multiple times (once per side effect).
    This is the key question for determining whether the task is multilabel.

    We use the first two columns as drug identifiers — verify this from
    the column names printed above before drawing conclusions.
    """
    _header("DRUG PAIR ANALYSIS")

    if df.shape[1] < 2:
        print("Too few columns to analyse drug pairs.")
        return

    col1, col2 = df.columns[0], df.columns[1]
    n_rows = len(df)
    n_unique_pairs = df[[col1, col2]].drop_duplicates().shape[0]
    avg_per_pair = n_rows / n_unique_pairs if n_unique_pairs > 0 else float("nan")

    print(f"  Drug column 1 : {col1!r}")
    print(f"  Drug column 2 : {col2!r}")
    print(f"  Total rows    : {n_rows:,}")
    print(f"  Unique pairs  : {n_unique_pairs:,}")
    print(f"  Rows per pair (avg) : {avg_per_pair:.1f}")

    if n_rows > n_unique_pairs:
        print()
        print("  → Each drug pair appears MORE THAN ONCE.")
        print("    This likely means one row = one (drug, drug, side-effect) triple.")
        print("    The task is probably multilabel: predict all side effects for a pair.")
    else:
        print()
        print("  → Each drug pair appears exactly once.")
        print("    The task structure is different — inspect the columns carefully.")


# ---------------------------------------------------------------------------
# Step 7 — Random sample for illustration
# ---------------------------------------------------------------------------

def print_random_sample(df: pd.DataFrame, n: int = 5) -> None:
    """Print a few random rows to see variety beyond the first rows."""
    _header(f"RANDOM SAMPLE ({n} ROWS)")
    sample = df.sample(n, random_state=RANDOM_SEED)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.max_colwidth", 50,
    ):
        print(sample.to_string(index=False))


# ---------------------------------------------------------------------------
# Step 8 — Save processed sample
# ---------------------------------------------------------------------------

def save_sample(df: pd.DataFrame, out_path: Path, n: int = SAMPLE_SIZE) -> None:
    """
    Save a small reproducible sample to data/processed/.
    Useful for quick iterative checks without reloading the full file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = df.sample(min(n, len(df)), random_state=RANDOM_SEED)
    sample.to_csv(out_path, index=False)
    print(f"\nSample of {len(sample):,} rows saved to: {out_path}")


# ---------------------------------------------------------------------------
# Step 9 — Open questions reminder
# ---------------------------------------------------------------------------

def print_open_questions() -> None:
    """
    Remind ourselves what we still need to decide before any modelling.
    Fill answers into notes/dataset_observations.md after reading the output.
    """
    _header("OPEN QUESTIONS — answer in notes/dataset_observations.md")
    questions = [
        "Are drugs identified by name or by an ID (e.g. DrugBank CID)?",
        "Are side effects identified by name or by a concept ID (e.g. UMLS CUI)?",
        "Does one row represent one (drug, drug, side-effect) triple?",
        "Are negative examples present, or only positive associations?",
        "Does the same drug pair appear multiple times (once per side effect)?",
        "Is the task binary, multiclass, or multilabel?",
        "Are drug or side-effect names available as text features, or only IDs?",
        "How many unique side effects are there?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_dataset(RAW_PATH)

    print_basic_info(df)
    print_missing_values(df)
    print_first_rows(df)
    print_unique_counts(df)
    print_drug_pair_analysis(df)
    print_random_sample(df)
    save_sample(df, SAMPLE_OUT)
    print_open_questions()

    print("\nDone. Read the output above and fill in notes/dataset_observations.md.\n")


if __name__ == "__main__":
    main()
