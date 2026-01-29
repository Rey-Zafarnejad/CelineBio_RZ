import argparse
import logging
from pathlib import Path

from immunofoundation.clues import preprocess_clues
from immunofoundation.config import load_config
from immunofoundation.foundation import train_foundation_model
from immunofoundation.logging_utils import setup_logging
from immunofoundation.report import make_clinic_report
from immunofoundation.rotterdam import build_rotterdam_reference
from immunofoundation.scoring import score_senescence

LOGGER = logging.getLogger(__name__)


def _resolve_output(config, subdir: str) -> Path:
    output_dir = config.output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_rotterdam(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(Path(config.raw["output"].get("log_path", config.output_dir / "pipeline.log")))
    build_rotterdam_reference(config.raw["rotterdam"], _resolve_output(config, "rotterdam"))


def run_clues(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(Path(config.raw["output"].get("log_path", config.output_dir / "pipeline.log")))
    preprocess_clues(config.raw["clues"], _resolve_output(config, "clues"))


def run_scoring(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(Path(config.raw["output"].get("log_path", config.output_dir / "pipeline.log")))
    score_senescence(config.raw["scoring"], _resolve_output(config, "scoring"))


def run_training(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(Path(config.raw["output"].get("log_path", config.output_dir / "pipeline.log")))
    train_foundation_model(config.raw["foundation_model"], _resolve_output(config, "foundation"))


def run_report(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    setup_logging(Path(config.raw["output"].get("log_path", config.output_dir / "pipeline.log")))
    make_clinic_report(config.raw["clinic_report"], _resolve_output(config, "reports"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ImmunoFoundationModel CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-rotterdam-reference").set_defaults(func=build_rotterdam)
    subparsers.add_parser("preprocess-clues").set_defaults(func=run_clues)
    subparsers.add_parser("score-senescence").set_defaults(func=run_scoring)
    subparsers.add_parser("train-foundation-model").set_defaults(func=run_training)
    subparsers.add_parser("make-clinic-report").set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
