import logging
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


def map_probes_to_symbols(probe_ids: Iterable[str]) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except ImportError as exc:
        raise ImportError("rpy2 is required for probe mapping.") from exc

    pandas2ri.activate()
    ro.r("suppressPackageStartupMessages(library(illuminaHumanv4.db))")
    probes = ro.StrVector(list(probe_ids))
    ro.globalenv["probe_ids"] = probes
    ro.r(
        "mapping <- AnnotationDbi::select(illuminaHumanv4.db, "
        "keys=probe_ids, columns=c('SYMBOL'), keytype='PROBEID')"
    )
    mapping = ro.r("mapping")
    df = pandas2ri.rpy2py(mapping)
    df = df.rename(columns={"PROBEID": "probe_id", "SYMBOL": "gene_symbol"})
    return df


def run_gsva(expr_gene: pd.DataFrame, gene_sets_path: str) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except ImportError as exc:
        raise ImportError("rpy2 is required for GSVA.") from exc

    pandas2ri.activate()
    ro.r("suppressPackageStartupMessages(library(GSVA))")
    ro.r("suppressPackageStartupMessages(library(GSEABase))")

    expr = expr_gene.set_index("gene")
    ro.globalenv["expr_matrix"] = pandas2ri.py2rpy(expr)
    ro.globalenv["gene_sets_path"] = gene_sets_path
    ro.r("gene_sets <- GSEABase::getGmt(gene_sets_path)")
    ro.r(
        "gsva_scores <- GSVA::gsva(as.matrix(expr_matrix), gene_sets, "
        "method='ssgsea', ssgsea.norm=TRUE, verbose=FALSE)"
    )
    scores = ro.r("gsva_scores")
    scores_df = pandas2ri.rpy2py(scores)
    scores_df.index.name = "gene_set"
    scores_df = scores_df.T.reset_index().rename(columns={"index": "sample_id"})
    return scores_df


def run_edger_tmm_logcpm(counts: pd.DataFrame) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except ImportError as exc:
        raise ImportError("rpy2 is required for edgeR TMM.") from exc

    pandas2ri.activate()
    ro.r("suppressPackageStartupMessages(library(edgeR))")
    ro.globalenv["counts"] = pandas2ri.py2rpy(counts)
    ro.r("dge <- DGEList(counts=as.matrix(counts))")
    ro.r("dge <- calcNormFactors(dge, method='TMM')")
    ro.r("logcpm <- cpm(dge, log=TRUE, prior.count=1)")
    logcpm = pandas2ri.rpy2py(ro.r("logcpm"))
    logcpm.index.name = "gene"
    return logcpm
