"""Shared input preprocessing for CSV and converted TSV trajectories."""

import pandas as pd
from scipy.signal import savgol_filter


PRESSURE_COLUMNS = ["Pano_in", "Pcat_in"]
PRESSURE_FILTER_WINDOW = 60
PRESSURE_FILTER_POLYORDER = 3
ARB_SPEED_COLUMN = "Com_nP1ActSpd"


def smooth_pressure(series: pd.Series) -> pd.Series:
    pressure = pd.to_numeric(series, errors="coerce")
    if pressure.isna().all():
        return pressure

    filled_pressure = pressure.interpolate(limit_direction="both")
    window_length = min(PRESSURE_FILTER_WINDOW, len(filled_pressure))
    if window_length <= PRESSURE_FILTER_POLYORDER:
        return filled_pressure

    try:
        smoothed = savgol_filter(
            filled_pressure.to_numpy(),
            window_length=window_length,
            polyorder=PRESSURE_FILTER_POLYORDER,
        )
    except ValueError:
        if window_length % 2 == 0 and window_length - 1 > PRESSURE_FILTER_POLYORDER:
            smoothed = savgol_filter(
                filled_pressure.to_numpy(),
                window_length=window_length - 1,
                polyorder=PRESSURE_FILTER_POLYORDER,
            )
        else:
            raise
    return pd.Series(smoothed, index=series.index)


def clean_stc(converted: pd.DataFrame, column: str) -> pd.Series:
    stc = converted[column].astype(float).copy()
    current = converted["I"].astype(float)

    current_zero_mask = current == 0
    active_zero_stc_mask = (current != 0) & (stc == 0)
    previous_valid_stc = stc.mask(current_zero_mask | active_zero_stc_mask).ffill()

    stc = stc.mask(active_zero_stc_mask, previous_valid_stc).fillna(stc)
    stc[current_zero_mask] = 10
    if column == "STC_ano" and ARB_SPEED_COLUMN in converted.columns:
        arb_speed = pd.to_numeric(converted[ARB_SPEED_COLUMN], errors="coerce")
        stc[arb_speed == 0] = 0
    return stc


def preprocess_trajectory_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean inputs before resampling, preserving outputs and metadata."""
    preprocessed = df.copy()
    for column in [*PRESSURE_COLUMNS, "I", "STC_ano", "STC_cat"]:
        if column in preprocessed.columns:
            preprocessed[column] = pd.to_numeric(preprocessed[column], errors="coerce")

    for column in PRESSURE_COLUMNS:
        if column in preprocessed.columns:
            preprocessed[column] = smooth_pressure(preprocessed[column]).fillna(preprocessed[column])

    if "I" in preprocessed.columns:
        preprocessed["I"] = preprocessed["I"].clip(lower=0)
        for column in ["STC_ano", "STC_cat"]:
            if column in preprocessed.columns:
                preprocessed[column] = clean_stc(preprocessed, column)
    return preprocessed
