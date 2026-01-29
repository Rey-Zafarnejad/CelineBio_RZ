import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from immunofoundation.io_utils import ensure_path
from immunofoundation.manifest import build_manifest, write_manifest

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path
    model_card_path: Path
    manifest_path: Path


def _load_datasets(dataset_configs: List[Dict[str, str]]) -> List[Dict[str, pd.DataFrame]]:
    datasets = []
    for dataset in dataset_configs:
        expr_path = ensure_path(dataset["expression_path"], "foundation.dataset.expression_path")
        meta_path = ensure_path(dataset["metadata_path"], "foundation.dataset.metadata_path")
        expr = pd.read_parquet(expr_path)
        meta = pd.read_parquet(meta_path)
        datasets.append({"expr": expr, "meta": meta, "name": dataset.get("name", expr_path.stem)})
    return datasets


def _prepare_training_matrix(expr: pd.DataFrame) -> np.ndarray:
    values = expr.drop(columns=["gene"]).T
    return values.astype(np.float32)


def train_foundation_model(config: Dict[str, object], output_dir: Path) -> TrainingArtifacts:
    datasets_config = config.get("datasets")
    if not datasets_config:
        LOGGER.warning("No training datasets provided. Skipping foundation model training.")
        raise RuntimeError("Training skipped: provide foundation_model.datasets in config.")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training. Install with pip install torch.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _load_datasets(datasets_config)

    matrices = []
    for dataset in datasets:
        matrix = _prepare_training_matrix(dataset["expr"])
        matrices.append(matrix)

    train_matrix = np.concatenate(matrices, axis=0)
    input_dim = train_matrix.shape[1]

    model_cfg = config.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    layers = int(model_cfg.get("layers", 2))

    class MaskedAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, layers: int):
            super().__init__()
            mods = []
            dim = input_dim
            for _ in range(layers):
                mods.append(nn.Linear(dim, hidden_dim))
                mods.append(nn.ReLU())
                dim = hidden_dim
            mods.append(nn.Linear(dim, input_dim))
            self.net = nn.Sequential(*mods)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = MaskedAutoencoder(input_dim, hidden_dim, layers)

    mask_ratio = float(config.get("mask_ratio", 0.15))
    batch_size = int(config.get("batch_size", 64))
    epochs = int(config.get("epochs", 10))

    dataset_tensor = torch.tensor(train_matrix)
    loader = DataLoader(TensorDataset(dataset_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            mask = torch.rand_like(batch) < mask_ratio
            masked = batch.clone()
            masked[mask] = 0
            preds = model(masked)
            loss = loss_fn(preds[mask], batch[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        LOGGER.info("Epoch %d loss %.4f", epoch + 1, epoch_loss / len(loader))

    model_path = output_dir / "foundation_model.pt"
    torch.save(model.state_dict(), model_path)

    metrics = {"training_loss": epoch_loss / len(loader)}
    metrics_path = output_dir / "foundation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    model_card_path = output_dir / "foundation_model_card.md"
    model_card_path.write_text(
        "# ImmunoFoundationModel Card\n\n"
        "This model was trained using masked gene reconstruction.\n"
        "No clinical labels were used.\n",
        encoding="utf-8",
    )

    manifest_path = output_dir / "foundation_manifest.json"
    write_manifest(build_manifest({"foundation_model": config}), manifest_path)

    return TrainingArtifacts(model_path, metrics_path, model_card_path, manifest_path)
