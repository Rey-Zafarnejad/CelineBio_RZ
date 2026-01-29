# ImmunoFoundationModel

Production-ready pipeline for reproductive health transcriptomics (reproductive aging, egg quality proxies, endometriosis risk).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional dependencies:
- `pip install -e .[rpy2]` for GSVA and edgeR interoperability.
- `pip install -e .[training]` for PyTorch-based foundation model training.

## Configuration

All paths are supplied through YAML. See `configs/example.yaml` and replace with real file paths.

### Required inputs
- Rotterdam metadata TSV.gz and expression TSV.gz.
- Probe-to-gene mapping CSV if rpy2 is unavailable.
- CLUES counts CSV.gz and metadata CSV.
- Ensembl-to-gene-symbol mapping CSV unless online mapping is enabled.
- Gene set file in GMT or CSV format for senescence signatures.

## CLI Usage

### Build Rotterdam reference
```bash
immunofoundation --config configs/example.yaml build-rotterdam-reference
```
Outputs:
- `rotterdam_expr_gene.parquet`
- `rotterdam_meta.parquet`
- `rotterdam_reference_gsva.joblib`
- `rotterdam_reference_pca.joblib`
- `rotterdam_manifest.json`

### Preprocess CLUES
```bash
immunofoundation --config configs/example.yaml preprocess-clues
```
Outputs:
- `clues_expr_gene.parquet`
- `clues_meta.parquet`
- `clues_manifest.json`

### Score senescence
```bash
immunofoundation --config configs/example.yaml score-senescence
```
Outputs:
- `senescence_scores.parquet`
- `senescence_manifest.json`

### Train foundation model
```bash
immunofoundation --config configs/example.yaml train-foundation-model
```
Training only runs when `foundation_model.datasets` contains real dataset paths. If none are provided, the command exits with a clear error.

Outputs:
- `foundation_model.pt`
- `foundation_metrics.json`
- `foundation_model_card.md`
- `foundation_manifest.json`

### Make clinic report
```bash
immunofoundation --config configs/example.yaml make-clinic-report
```
Outputs:
- `clinic_reports/*.json`
- `clinic_report_table.csv`
- `clinic_report_manifest.json`

## Adding new datasets

1. Provide expression matrices with `gene` column and sample columns.
2. Provide metadata with `sample_id`, `age`, `sex`, and optional labels.
3. Update `foundation_model.datasets` entries in the config:

```yaml
foundation_model:
  datasets:
    - name: cohort_a
      expression_path: /path/to/expr.parquet
      metadata_path: /path/to/meta.parquet
```

## Notes

- If required files are missing, the pipeline raises a clear error describing the missing path.
- All runs emit JSON manifests capturing configuration, package versions, and git commit.
