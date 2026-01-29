import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from immunofoundation.io_utils import ensure_path, write_parquet
from immunofoundation.manifest import build_manifest, write_manifest

LOGGER = logging.getLogger(__name__)


def _encode_sex(series: pd.Series) -> pd.Series:
    sex = series.astype(str).str.lower()
    return sex.map({"male": 1, "m": 1, "female": 0, "f": 0}).fillna(0)


def _prepare_expression(expr_path: Path) -> pd.DataFrame:
    expr = pd.read_parquet(expr_path)
    if "gene" not in expr.columns:
        raise ValueError("Expression parquet must include gene column.")
    return expr


def _gsva_scores(expr_gene: pd.DataFrame, gene_sets_path: str, use_rpy2: bool) -> pd.DataFrame:
    if not use_rpy2:
        raise RuntimeError("GSVA scoring requires rpy2 with GSVA installed.")
    try:
        from immunofoundation.rpy2_utils import run_gsva
    except ImportError as exc:
        raise RuntimeError("rpy2 requested but not available. Install rpy2.") from exc
    return run_gsva(expr_gene, gene_sets_path)


def _project_pca(expr_gene: pd.DataFrame, pca_ref: Dict[str, object]) -> pd.DataFrame:
    expr = expr_gene.set_index("gene")
    scores = {}
    for gene_set, model in pca_ref["pca_models"].items():
        genes = model["genes"]
        subset = expr.reindex(genes).T
        if subset.isna().any().any():
            subset = subset.fillna(0)
        pca = model["pca"]
        pc_idx = model["pc_index"]
        pca_scores = pca.transform(subset)[:, pc_idx]
        scores[gene_set] = pca_scores
    return pd.DataFrame(scores, index=subset.index)


def score_senescence(config: Dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    expr_path = ensure_path(config["expression_path"], "scoring.expression_path")
    meta_path = ensure_path(config["metadata_path"], "scoring.metadata_path")
    gsva_ref = ensure_path(config["gsva_reference_path"], "scoring.gsva_reference_path")
    pca_ref_path = ensure_path(config["pca_reference_path"], "scoring.pca_reference_path")

    expr_gene = _prepare_expression(expr_path)
    meta = pd.read_parquet(meta_path)

    gsva_ref_obj = joblib.load(gsva_ref)
    gene_sets_path = config.get("gene_sets_path") or gsva_ref_obj.get("gene_sets_path")
    if not gene_sets_path:
        raise FileNotFoundError("gene_sets_path required for GSVA scoring.")

    gsva_scores = _gsva_scores(expr_gene, gene_sets_path, config.get("use_rpy2", False))
    gsva_scores = gsva_scores.set_index("sample_id")

    pca_ref = joblib.load(pca_ref_path)
    pca_scores = _project_pca(expr_gene, pca_ref)

    output_records = []
    for method_name, scores in {"gsva": gsva_scores, "pca": pca_scores}.items():
        for gene_set in scores.columns:
            model = (
                gsva_ref_obj["models"][gene_set]
                if method_name == "gsva"
                else pca_ref["lin_models"][gene_set]
            )
            residual_stats = (
                gsva_ref_obj["residual_stats"][gene_set]
                if method_name == "gsva"
                else pca_ref["residual_stats"][gene_set]
            )
            X = pd.DataFrame(
                {
                    "age": meta[config["age_col"]],
                    "sex": _encode_sex(meta[config["sex_col"]]),
                },
                index=meta[config["sample_id_col"]].astype(str),
            )
            preds = model.predict(X)
            accel = scores[gene_set].loc[X.index] - preds
            accel_z = (accel - residual_stats["mean"]) / residual_stats["std"]
            for sample_id in X.index:
                output_records.append(
                    {
                        "sample_id": sample_id,
                        "method": method_name,
                        "gene_set": gene_set,
                        "score": float(scores.loc[sample_id, gene_set]),
                        "expected": float(preds[X.index.get_loc(sample_id)]),
                        "accel": float(accel.loc[sample_id]),
                        "accel_z_ref": float(accel_z.loc[sample_id]),
                    }
                )

    output_df = pd.DataFrame(output_records)
    output_path = output_dir / "senescence_scores.parquet"
    write_parquet(output_df, output_path)

    manifest_path = output_dir / "senescence_manifest.json"
    write_manifest(build_manifest({"scoring": config}), manifest_path)
    return output_path
