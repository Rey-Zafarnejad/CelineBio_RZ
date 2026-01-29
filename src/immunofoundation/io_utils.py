from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "mapping": ["probe_id", "gene_symbol"],
    "gene_sets": ["gene_set", "gene_symbol"],
}


def ensure_path(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Required {description} file is missing: {resolved}. "
            f"Provide it in the config under {description}."
        )
    return resolved


def read_table(path: Path, sep: str = "\t") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def validate_columns(df: pd.DataFrame, expected: Iterable[str], context: str) -> None:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}")


def validate_no_duplicate_samples(meta: pd.DataFrame, sample_id_col: str) -> None:
    if meta[sample_id_col].duplicated().any():
        dupes = meta.loc[meta[sample_id_col].duplicated(), sample_id_col].unique()
        raise ValueError(f"Duplicate sample IDs detected: {dupes.tolist()}")


def validate_mapping_success(mapping: pd.DataFrame, id_col: str, mapped_col: str, min_rate: float) -> None:
    total = mapping[id_col].nunique()
    mapped = mapping.dropna(subset=[mapped_col])[id_col].nunique()
    rate = mapped / total if total else 0
    if rate < min_rate:
        raise ValueError(
            f"Mapping success rate {rate:.2%} below required {min_rate:.2%}."
        )
