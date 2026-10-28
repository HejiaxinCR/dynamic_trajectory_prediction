"""
Convert exported PEMFC data to trajectory CSV format and preprocess inputs.

Usage:
    python interpolate_stc.py --input "data/to wenwen.tsv" --output data/to_wenwen_converted.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from trajectory_preprocessing import ARB_SPEED_COLUMN, preprocess_trajectory_dataset


CURRENTS = np.array([43.4, 83.1, 120.1, 160.2, 210.4, 250.3, 300.8, 350.7, 411.1, 450.1, 480.7])
DATA_OPCON2_0 = {
    1.9: [2.90, 2.39, 2.19, 2.12, 2.06, 2.02, 2.02, 1.98, 1.89, 1.84, 1.80],
    2.2: [3.31, 2.64, 2.38, 2.30, 2.22, 2.16, 2.16, 2.08, 1.99, 1.91, 1.87],
    2.4: [3.42, 2.79, 2.50, 2.41, 2.32, 2.26, 2.26, 2.17, 2.06, 1.99, 1.94],
}

CURRENT_COL = "FuCellElec_iStckAct"
ARB_SPEED_COL = ARB_SPEED_COLUMN
ARB_SPEED_SCALE = 1e4
RAW_PRESSURE_SOURCES = {
    "Pano_in": [r"PAndeDs_p_f\XETK:1", "PAndeUs_p_f"],
    "Pcat_in": ["PCtdeUs_p_f"],
}
OUTPUT_COL = "STC_ano"
TARGET_COLUMNS = [
    "t",
    "Tano_in",
    "RHano_in",
    "Pano_in",
    "STC_ano",
    "XH2_dry",
    "Tcat_in",
    "RHcat_in",
    "Pcat_in",
    "STC_cat",
    "TCL_in",
    "FR_CL",
    "I",
    "sa",
    "ua",
    "lambda_airin",
]
SOURCE_TO_TARGET = {
    "time": "t",
    "TAndeUs_t_f": "Tano_in",
    "PAndeUs_p_f": "Pano_in",
    "TCtdeUs_t_f": "Tcat_in",
    "RH_ctde_in": "RHcat_in",
    "PCtdeUs_p_f": "Pcat_in",
    "TCooltStckIn_t_f": "TCL_in",
    "HiTPmp_mfCooltMeasd": "FR_CL",
    CURRENT_COL: "I",
    "FuCellAir_ratLamCtdeAct": "STC_cat",
}
REQUIRED_SOURCE_COLUMNS = list(dict.fromkeys([*SOURCE_TO_TARGET.keys(), ARB_SPEED_COL]))
REQUIRED_TRAJECTORY_COLUMNS = TARGET_COLUMNS.copy()
REQUIRED_TRAJECTORY_INPUT_COLUMNS = [
    "t",
    "Tano_in",
    "RHano_in",
    "Pano_in",
    "STC_ano",
    "XH2_dry",
    "Tcat_in",
    "RHcat_in",
    "Pcat_in",
    "STC_cat",
    "TCL_in",
    "FR_CL",
    "I",
]


def build_known_points():
    """Expand calibration curves to (current, ARB speed) -> STC points."""
    points, values = [], []
    for arb_speed, stoich_values in DATA_OPCON2_0.items():
        for current_val, stoich in zip(CURRENTS, stoich_values):
            points.append([current_val, arb_speed])
            values.append(stoich)
    return np.array(points), np.array(values)


def safe_griddata(known_points, known_values, target_points):
    """
    Interpolate with cubic first, then fill out-of-hull points with nearest.
    """
    interpolated_values = griddata(known_points, known_values, target_points, method="cubic")
    nan_indices = np.isnan(interpolated_values)
    if np.any(nan_indices):
        nearest_values = griddata(known_points, known_values, target_points[nan_indices], method="nearest")
        interpolated_values[nan_indices] = nearest_values
    return interpolated_values


def apply_raw_pressure_source(df: pd.DataFrame, converted: pd.DataFrame) -> None:
    for target_col, source_cols in RAW_PRESSURE_SOURCES.items():
        for source_col in source_cols:
            if source_col in df.columns:
                converted[target_col] = df[source_col]
                break


def read_input(input_path: str) -> pd.DataFrame:
    separator = "\t" if Path(input_path).suffix.lower() == ".tsv" else ","
    return pd.read_csv(input_path, sep=separator, low_memory=False)


def normalize_blank_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        blank_mask = df[column].astype("string").str.fullmatch(r"\s*", na=False)
        df.loc[blank_mask, column] = np.nan
    return df


def drop_missing_input_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_blank_strings(df)
    return df.dropna(subset=REQUIRED_SOURCE_COLUMNS).reset_index(drop=True)


def drop_missing_trajectory_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_blank_strings(df)
    df = coerce_numeric_columns(df, TARGET_COLUMNS)
    return df.dropna(subset=REQUIRED_TRAJECTORY_INPUT_COLUMNS).reset_index(drop=True)


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return set(columns).issubset(df.columns)


def detect_input_format(df: pd.DataFrame) -> str:
    if has_columns(df, REQUIRED_SOURCE_COLUMNS):
        return "raw"
    if has_columns(df, REQUIRED_TRAJECTORY_COLUMNS):
        return "trajectory"
    missing_raw = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(df.columns))
    missing_trajectory = sorted(set(REQUIRED_TRAJECTORY_COLUMNS) - set(df.columns))
    raise ValueError(
        "Input is neither raw export format nor trajectory format. "
        f"Missing raw columns: {missing_raw}; missing trajectory columns: {missing_trajectory}"
    )


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def interpolate_stc_ano(df: pd.DataFrame) -> np.ndarray:
    points, values = build_known_points()
    df = df.copy()
    df.loc[df[CURRENT_COL] < 0, CURRENT_COL] = 0

    valid_mask = df[[ARB_SPEED_COL, CURRENT_COL]].notna().all(axis=1)
    stc = np.full(len(df), np.nan, dtype=float)

    if not valid_mask.any():
        return stc

    valid_df = df.loc[valid_mask]
    current_target = np.array(valid_df[CURRENT_COL])
    arb_target = np.array(valid_df[ARB_SPEED_COL]) / ARB_SPEED_SCALE
    target_points = np.vstack((current_target, arb_target)).T

    valid_stc = safe_griddata(points, values, target_points)

    stc[valid_mask.to_numpy()] = valid_stc
    return stc


def convert_raw_to_trajectory_format(df: pd.DataFrame) -> pd.DataFrame:
    converted = pd.DataFrame(index=df.index)
    for source_col, target_col in SOURCE_TO_TARGET.items():
        converted[target_col] = df[source_col]

    apply_raw_pressure_source(df, converted)

    converted["Pano_in"] = converted["Pano_in"] / 1000
    converted["Pcat_in"] = converted["Pcat_in"] / 1000
    converted["I"] = df[CURRENT_COL]
    converted[OUTPUT_COL] = interpolate_stc_ano(df)
    converted["RHano_in"] = 0.65
    converted["XH2_dry"] = 0.85
    converted["sa"] = np.nan
    converted["ua"] = np.nan
    converted["lambda_airin"] = np.nan
    # Keep the source speed so shared preprocessing can retain the stop rule.
    converted[ARB_SPEED_COL] = df[ARB_SPEED_COL]

    return converted[[*TARGET_COLUMNS, ARB_SPEED_COL]]


def run(input_path: str, output_path: str, *, preprocess: bool = True) -> pd.DataFrame:
    df = read_input(input_path)
    input_format = detect_input_format(df)

    if input_format == "raw":
        df = drop_missing_input_rows(df)
        trajectory = convert_raw_to_trajectory_format(df)
    else:
        trajectory = drop_missing_trajectory_rows(df)

    result = preprocess_trajectory_dataset(trajectory) if preprocess else trajectory
    result.to_csv(output_path, index=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input TSV or CSV path")
    parser.add_argument("--output", required=True, help="Converted CSV output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_df = run(args.input, args.output)
    print(f"Saved preprocessed trajectory CSV to: {args.output}, rows: {len(result_df)}")
