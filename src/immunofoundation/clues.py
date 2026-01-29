import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from immunofoundation.io_utils import (
    ensure_path,
    validate_mapping_success,
    validate_no_duplicate_samples,
    write_parquet,
)
from immunofoundation.manifest import build_manifest, write_manifest

LOGGER = logging.getLogger(__name__)


@dataclass
class CluesArtifacts:
    expr_path: Path
    meta_path: Path
    manifest_path: Path


def _read_counts(path: Path) -> pd.DataFrame:
    counts = pd.read_csv(path)
    if counts.columns[0].lower() not in {"gene", "gene_id", "ensembl_id"}:
        counts = counts.rename(columns={counts.columns[0]: "gene_id"})
    return counts


def _library_size_qc(counts: pd.DataFrame, min_lib_size: int) -> pd.DataFrame:
    lib_sizes = counts.drop(columns=["gene_id"]).sum(axis=0)
    keep = lib_sizes[lib_sizes >= min_lib_size].index
    removed = set(lib_sizes.index) - set(keep)
    if removed:
        LOGGER.warning("Removed %d samples due to library size QC", len(removed))
    return counts[["gene_id"] + list(keep)]


def _filter_cpm(counts: pd.DataFrame, min_cpm: float, min_samples: int) -> pd.DataFrame:
    lib_sizes = counts.drop(columns=["gene_id"]).sum(axis=0)
    cpm = counts.drop(columns=["gene_id"]).div(lib_sizes, axis=1) * 1e6
    mask = (cpm >= min_cpm).sum(axis=1) >= min_samples
    filtered = counts.loc[mask].reset_index(drop=True)
    LOGGER.info("Filtered genes: %d -> %d", counts.shape[0], filtered.shape[0])
    return filtered


def _normalize_logcpm(counts: pd.DataFrame, use_rpy2: bool) -> pd.DataFrame:
    counts_matrix = counts.set_index("gene_id")
    if use_rpy2:
        try:
            from immunofoundation.rpy2_utils import run_edger_tmm_logcpm
        except ImportError as exc:
            raise RuntimeError("rpy2 requested but not available. Install rpy2.") from exc
        logcpm = run_edger_tmm_logcpm(counts_matrix)
    else:
        LOGGER.warning(
            "Using size-factor normalization (DESeq-like) because rpy2/edgeR not available."
        )
        geometric_means = np.exp(np.log(counts_matrix.replace(0, np.nan)).mean(axis=1))
        ratios = counts_matrix.div(geometric_means, axis=0)
        size_factors = ratios.median(axis=0).replace(0, np.nan)
        norm_counts = counts_matrix.div(size_factors, axis=1)
        logcpm = np.log2(norm_counts.div(norm_counts.sum(axis=0), axis=1) * 1e6 + 1)
    logcpm = logcpm.reset_index().rename(columns={"gene_id": "gene"})
    return logcpm


def _map_ensembl_to_symbol(config: Dict[str, str], genes: pd.Series) -> pd.DataFrame:
    mapping_path = config.get("gene_mapping_path")
    if mapping_path:
        mapping = pd.read_csv(ensure_path(mapping_path, "clues.gene_mapping_path"))
        mapping.columns = [col.lower() for col in mapping.columns]
        if "gene_id" not in mapping.columns or "gene_symbol" not in mapping.columns:
            raise ValueError("Mapping file must include gene_id and gene_symbol columns.")
        mapping = mapping.rename(columns={"gene_symbol": "gene"})
        return mapping[["gene_id", "gene"]]
    if config.get("allow_online_mapping"):
        try:
            import mygene
        except ImportError as exc:
            raise RuntimeError("mygene is required for online mapping.") from exc
        mg = mygene.MyGeneInfo()
        query = mg.querymany(
            genes.tolist(),
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
        )
        records = [
            {"gene_id": item["query"], "gene": item.get("symbol")}
            for item in query
        ]
        return pd.DataFrame(records)
    raise FileNotFoundError(
        "Ensembl-to-symbol mapping missing. Provide clues.gene_mapping_path or enable online mapping."
    )


def _collapse_symbols(expr: pd.DataFrame, method: str) -> pd.DataFrame:
    value_cols = [col for col in expr.columns if col != "gene"]
    if method == "median":
        agg = "median"
    else:
        agg = "mean"
    collapsed = expr.groupby("gene", as_index=False)[value_cols].agg(agg)
    return collapsed


def preprocess_clues(config: Dict[str, str], output_dir: Path) -> CluesArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = ensure_path(config["counts_path"], "clues.counts_path")
    counts = _read_counts(counts_path)
    counts = _library_size_qc(counts, config["min_library_size"])
    counts = _filter_cpm(
        counts,
        config.get("min_cpm", 0.5),
        config.get("min_samples", max(1, int(counts.shape[1] * 0.1))),
    )
    logcpm = _normalize_logcpm(counts, config.get("use_rpy2", False))

    mapping = _map_ensembl_to_symbol(config, logcpm["gene"])
    validate_mapping_success(mapping, "gene_id", "gene", config.get("min_mapping_rate", 0.8))
    expr = logcpm.merge(mapping, left_on="gene", right_on="gene_id", how="left")
    missing = expr["gene"].isna().sum()
    if missing:
        LOGGER.warning("Missing gene symbol for %d Ensembl IDs", missing)
    expr = expr.dropna(subset=["gene"])
    expr = expr.drop(columns=["gene_id"])

    expr = _collapse_symbols(expr, config.get("collapse_method", "mean"))

    if config.get("zscore", True):
        values = expr.drop(columns=["gene"])
        zvals = (values - values.mean(axis=1).values[:, None]) / values.std(axis=1).values[:, None]
        expr = pd.concat([expr[["gene"]], pd.DataFrame(zvals, columns=values.columns)], axis=1)

    expr_out = output_dir / "clues_expr_gene.parquet"
    meta_out = output_dir / "clues_meta.parquet"

    write_parquet(expr, expr_out)

    meta_path = config.get("metadata_path")
    if not meta_path:
        raise FileNotFoundError("clues.metadata_path is required to build meta output.")
    meta = pd.read_csv(ensure_path(meta_path, "clues.metadata_path"))
    validate_no_duplicate_samples(meta, config.get("sample_id_col", "sample_id"))
    write_parquet(meta, meta_out)

    manifest_path = output_dir / "clues_manifest.json"
    write_manifest(build_manifest({"clues": config}), manifest_path)

    return CluesArtifacts(expr_out, meta_out, manifest_path)
