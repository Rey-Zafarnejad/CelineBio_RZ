import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from immunofoundation.io_utils import ensure_path, validate_mapping_success, validate_no_duplicate_samples, write_parquet
from immunofoundation.manifest import build_manifest, write_manifest

LOGGER = logging.getLogger(__name__)


@dataclass
class RotterdamArtifacts:
    expr_path: Path
    meta_path: Path
    gsva_path: Path
    pca_path: Path
    manifest_path: Path


def _load_metadata(path: Path, sample_id_col: str) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t")
    if sample_id_col not in meta.columns:
        raise ValueError(f"Metadata missing sample id column: {sample_id_col}")
    return meta


def _load_expression(path: Path) -> pd.DataFrame:
    expr = pd.read_csv(path, sep="\t")
    if expr.columns[0].lower() not in {"probe_id", "probe", "id"}:
        expr = expr.rename(columns={expr.columns[0]: "probe_id"})
    return expr


def _align_samples(expr: pd.DataFrame, meta: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
    sample_ids = meta[sample_id_col].astype(str)
    expr_columns = [col for col in expr.columns if col != "probe_id"]
    missing = set(sample_ids) - set(expr_columns)
    if missing:
        raise ValueError(f"Expression matrix missing {len(missing)} samples from metadata.")
    aligned = expr[["probe_id"] + list(sample_ids)]
    return aligned


def _load_probe_mapping(config: Dict[str, str], probes: Iterable[str]) -> pd.DataFrame:
    if config.get("use_rpy2"):
        try:
            from immunofoundation.rpy2_utils import map_probes_to_symbols
        except ImportError as exc:
            raise RuntimeError("rpy2 requested but not available. Install rpy2.") from exc
        return map_probes_to_symbols(list(probes))
    mapping_path = config.get("probe_mapping_path")
    if not mapping_path:
        raise FileNotFoundError(
            "Probe-to-gene mapping file required. Provide rotterdam.probe_mapping_path."
        )
    mapping_df = pd.read_csv(ensure_path(mapping_path, "rotterdam.probe_mapping_path"))
    mapping_df.columns = [col.lower() for col in mapping_df.columns]
    if "probe_id" not in mapping_df.columns or "gene_symbol" not in mapping_df.columns:
        raise ValueError("Mapping file must include probe_id and gene_symbol columns.")
    return mapping_df[["probe_id", "gene_symbol"]]


def _collapse_to_gene(expr: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = expr.merge(mapping, on="probe_id", how="left")
    missing = merged["gene_symbol"].isna().sum()
    if missing:
        LOGGER.warning("Missing gene symbols for %d probes", missing)
    merged = merged.dropna(subset=["gene_symbol"])
    value_cols = [col for col in merged.columns if col not in {"probe_id", "gene_symbol"}]
    collapsed = (
        merged.groupby("gene_symbol", as_index=False)[value_cols]
        .mean()
        .rename(columns={"gene_symbol": "gene"})
    )
    return collapsed


def build_rotterdam_reference(config: Dict[str, str], output_dir: Path) -> RotterdamArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ensure_path(config["metadata_path"], "rotterdam.metadata_path")
    expr_path = ensure_path(config["expression_path"], "rotterdam.expression_path")
    meta = _load_metadata(meta_path, config["sample_id_col"])
    validate_no_duplicate_samples(meta, config["sample_id_col"])
    expr = _load_expression(expr_path)
    expr_aligned = _align_samples(expr, meta, config["sample_id_col"])
    mapping = _load_probe_mapping(config, expr_aligned["probe_id"].tolist())
    validate_mapping_success(mapping, "probe_id", "gene_symbol", config.get("min_mapping_rate", 0.8))
    expr_gene = _collapse_to_gene(expr_aligned, mapping)

    expr_out = output_dir / "rotterdam_expr_gene.parquet"
    meta_out = output_dir / "rotterdam_meta.parquet"
    write_parquet(expr_gene, expr_out)
    write_parquet(meta, meta_out)

    gsva_path = output_dir / "rotterdam_reference_gsva.joblib"
    pca_path = output_dir / "rotterdam_reference_pca.joblib"
    manifest_path = output_dir / "rotterdam_manifest.json"

    build_rotterdam_gsva_reference(
        config,
        expr_gene,
        meta,
        gsva_path,
    )
    build_rotterdam_pca_reference(
        config,
        expr_gene,
        meta,
        pca_path,
    )

    write_manifest(build_manifest({"rotterdam": config}), manifest_path)

    return RotterdamArtifacts(expr_out, meta_out, gsva_path, pca_path, manifest_path)


def _encode_sex(meta: pd.DataFrame, sex_col: str) -> pd.Series:
    sex = meta[sex_col].astype(str).str.lower()
    return sex.map({"male": 1, "m": 1, "female": 0, "f": 0}).fillna(0)


def build_rotterdam_gsva_reference(
    config: Dict[str, str],
    expr_gene: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    if not config.get("use_rpy2"):
        raise RuntimeError("GSVA reference requires rpy2 with GSVA installed.")
    try:
        from immunofoundation.rpy2_utils import run_gsva
    except ImportError as exc:
        raise RuntimeError("rpy2 requested but not available. Install rpy2.") from exc

    gene_sets_path = ensure_path(config["gene_sets_path"], "rotterdam.gene_sets_path")
    scores = run_gsva(expr_gene, gene_sets_path)

    models = {}
    residual_stats = {}
    for gene_set in scores.columns:
        X = pd.DataFrame(
            {
                "age": meta[config["age_col"]],
                "sex": _encode_sex(meta, config["sex_col"]),
            }
        )
        y = scores[gene_set]
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        residual_stats[gene_set] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std(ddof=1)),
        }
        models[gene_set] = model

    joblib.dump(
        {
            "gene_sets_path": str(gene_sets_path),
            "scores": scores,
            "models": models,
            "residual_stats": residual_stats,
        },
        output_path,
    )


def build_rotterdam_pca_reference(
    config: Dict[str, str],
    expr_gene: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    gene_sets_path = ensure_path(config["gene_sets_path"], "rotterdam.gene_sets_path")
    gene_sets = pd.read_csv(gene_sets_path)
    if "gene_set" not in gene_sets.columns or "gene_symbol" not in gene_sets.columns:
        raise ValueError("Gene sets file must include gene_set and gene_symbol columns.")

    expr_matrix = expr_gene.set_index("gene")
    pca_models = {}
    residual_stats = {}
    lin_models = {}
    for gene_set, genes in gene_sets.groupby("gene_set"):
        gene_list = genes["gene_symbol"].unique().tolist()
        missing = [g for g in gene_list if g not in expr_matrix.index]
        if missing:
            LOGGER.warning("%s missing %d genes from expression matrix", gene_set, len(missing))
        subset = expr_matrix.loc[expr_matrix.index.intersection(gene_list)]
        if subset.empty:
            raise ValueError(f"No genes found for gene set {gene_set}")
        X = subset.T
        pca = PCA(n_components=min(5, X.shape[1]))
        scores = pca.fit_transform(X)
        age = meta[config["age_col"]].values
        correlations = [np.corrcoef(scores[:, idx], age)[0, 1] for idx in range(scores.shape[1])]
        best_idx = int(np.nanargmax(np.abs(correlations)))
        pc_scores = scores[:, best_idx]
        if correlations[best_idx] < 0:
            pc_scores = -pc_scores
            pca.components_[best_idx, :] *= -1
        sex = _encode_sex(meta, config["sex_col"])
        model = LinearRegression().fit(
            pd.DataFrame({"age": meta[config["age_col"]], "sex": sex}),
            pc_scores,
        )
        residuals = pc_scores - model.predict(
            pd.DataFrame({"age": meta[config["age_col"]], "sex": sex})
        )
        residual_stats[gene_set] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std(ddof=1)),
        }
        pca_models[gene_set] = {
            "pca": pca,
            "pc_index": best_idx,
            "genes": subset.index.tolist(),
        }
        lin_models[gene_set] = model

    joblib.dump(
        {
            "pca_models": pca_models,
            "lin_models": lin_models,
            "residual_stats": residual_stats,
            "gene_sets_path": str(gene_sets_path),
        },
        output_path,
    )
