import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pkg_resources


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def build_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "packages": {dist.project_name: dist.version for dist in pkg_resources.working_set},
        "config": config,
    }


def write_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
