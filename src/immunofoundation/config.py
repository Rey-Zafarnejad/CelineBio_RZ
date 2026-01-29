import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["output"]["dir"]).expanduser().resolve()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    return Config(raw=data)


def dump_config(config: Config, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config.raw, handle, indent=2)
