from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_gru_predictor import run_prediction


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "datasets"
DEFAULT_MODEL_DIR = ROOT_DIR / "model_weights"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "prediction_results"
DEFAULT_MODEL_CONFIG = DEFAULT_MODEL_DIR / "joint_gru_lambda_aux_training_config.json"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "joint_gru_lambda_aux_B_exogenous_u_only.pt"


def _discover_csvs(dataset_dir: Path) -> dict[str, Path]:
    paths = sorted(dataset_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV files were found in {dataset_dir}")
    return {path.stem: path for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic GRU free-run trajectory prediction.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Folder containing input trajectory CSV files.")
    parser.add_argument("--csv", type=Path, action="append", default=None, help="Specific CSV to predict. Can be passed multiple times.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Training config JSON exported by the notebook.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Trained GRU checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for prediction CSVs, metrics, checks, and plots.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows while saving plots.")
    parser.add_argument("--disable-hard-physical-gate", action="store_true", help="Do not force outputs to zero when current or stoichiometry is inactive.")
    parser.add_argument("--current-threshold", type=float, default=0.0, help="Force outputs to zero when I is at or below this threshold.")
    parser.add_argument("--stc-threshold", type=float, default=0.0, help="Force outputs to zero when STC_ano or STC_cat is at or below this threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv:
        data_paths = {path.stem: path for path in args.csv}
    else:
        data_paths = _discover_csvs(args.dataset_dir)

    prediction_df, metrics_df, checks_df = run_prediction(
        model_config_path=args.model_config,
        model_path=args.model_path,
        data_paths=data_paths,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        hard_physical_gate=not args.disable_hard_physical_gate,
        current_threshold=args.current_threshold,
        stc_threshold=args.stc_threshold,
    )
    print(f"Predicted trajectories: {prediction_df['trajectory_id'].nunique()}")
    print(f"Prediction rows: {len(prediction_df)}")
    print(f"Per-trajectory outputs: {args.output_dir}")
    print(f"Metrics rows: {len(metrics_df)}")
    print(f"Physical check rows: {len(checks_df)}")


if __name__ == "__main__":
    main()
