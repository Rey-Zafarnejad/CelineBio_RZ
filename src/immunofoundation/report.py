import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from immunofoundation.io_utils import ensure_path
from immunofoundation.manifest import build_manifest, write_manifest

LOGGER = logging.getLogger(__name__)


def make_clinic_report(config: Dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = ensure_path(config["scores_path"], "report.scores_path")
    scores = pd.read_parquet(scores_path)
    if scores.empty:
        raise ValueError("Senescence scores are empty.")

    report_dir = output_dir / "clinic_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    pdf_table_records = []
    for sample_id, group in scores.groupby("sample_id"):
        record = {"sample_id": sample_id}
        for _, row in group.iterrows():
            key = f"{row['method']}_{row['gene_set']}"
            record[f"{key}_accel_z"] = row["accel_z_ref"]
        record["explanation"] = "Top contributing genes require model weights or SHAP; not available."
        json_path = report_dir / f"{sample_id}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        pdf_table_records.append(record)

    pdf_table = pd.DataFrame(pdf_table_records)
    pdf_table_path = output_dir / "clinic_report_table.csv"
    pdf_table.to_csv(pdf_table_path, index=False)

    manifest_path = output_dir / "clinic_report_manifest.json"
    write_manifest(build_manifest({"clinic_report": config}), manifest_path)

    return pdf_table_path
