"""Download and validate the processed Cleveland Heart Disease data."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.zip"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleveland.csv"
ARCHIVE_MEMBER = "processed.cleveland.data"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a Cleveland dataset DataFrame without imputing it."""
    if list(data.columns) != COLUMNS:
        raise ValueError(
            f"Unexpected dataset columns. Expected {COLUMNS}, got {list(data.columns)}."
        )
    if data.empty:
        raise ValueError("The Cleveland dataset is empty.")

    normalized = data.copy()
    for column in COLUMNS:
        converted = pd.to_numeric(normalized[column], errors="coerce")
        invalid = normalized[column].notna() & converted.isna()
        if invalid.any():
            examples = normalized.loc[invalid, column].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"Unexpected non-numeric values in {column!r}: {examples}."
            )
        normalized[column] = converted

    if normalized["num"].isna().any():
        raise ValueError("The dataset contains missing or invalid target values in `num`.")
    target_values = set(normalized["num"].unique())
    if not target_values.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected raw target labels: {sorted(target_values)}.")
    return normalized


def _parse_archive(content: bytes) -> pd.DataFrame:
    """Read and validate the Cleveland member from UCI's ZIP archive bytes."""
    try:
        with ZipFile(BytesIO(content)) as archive:
            member = next(
                (
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(ARCHIVE_MEMBER)
                ),
                None,
            )
            if member is None:
                raise ValueError(
                    "The downloaded UCI archive does not contain "
                    f"`{ARCHIVE_MEMBER}`."
                )
            with archive.open(member) as data_file:
                data = pd.read_csv(
                    data_file,
                    header=None,
                    names=COLUMNS,
                    na_values="?",
                    skipinitialspace=True,
                )
    except BadZipFile as error:
        raise ValueError("The UCI data source was not a valid ZIP archive.") from error

    return validate_dataset(data)


def process_local_archive(
    archive_path: Path = ARCHIVE_PATH, data_path: Path = DATA_PATH
) -> pd.DataFrame:
    """Validate an already downloaded UCI ZIP and write the processed CSV."""
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Archive not found at {archive_path}. Download the UCI ZIP from "
            f"{DATA_URL} and save it there."
        )
    data = _parse_archive(archive_path.read_bytes())
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)
    return data


def download_dataset(
    archive_path: Path = ARCHIVE_PATH, data_path: Path = DATA_PATH
) -> pd.DataFrame:
    """Download the official UCI archive, extract Cleveland data, and save CSV."""
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    data = _parse_archive(response.content)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(response.content)
    data.to_csv(data_path, index=False)
    return data


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the processed local CSV."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python -m src.data_loader` first."
        )
    return validate_dataset(pd.read_csv(data_path))


def main() -> None:
    """Download the dataset and print factual validation details."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-archive",
        action="store_true",
        help=f"Use an existing ZIP at {ARCHIVE_PATH} without network access.",
    )
    args = parser.parse_args()
    if args.from_archive:
        data = process_local_archive()
        action = f"Processed existing UCI archive at {ARCHIVE_PATH}"
    else:
        data = download_dataset()
        action = "Downloaded official UCI Cleveland data"
    print(f"{action} to {DATA_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Raw target labels: {sorted(data['num'].unique().tolist())}")
    print(f"Missing values by column:\n{data.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
