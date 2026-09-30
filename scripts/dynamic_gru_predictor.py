from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


PLOT_YLABELS = {
    "Tano_in": "Temperature (degC)",
    "Tcat_in": "Temperature (degC)",
    "TCL_in": "Temperature (degC)",
    "RHano_in": "Relative humidity (-)",
    "RHcat_in": "Relative humidity (-)",
    "Pano_in": "Pressure (bar)",
    "Pcat_in": "Pressure (bar)",
    "STC_ano": "Stoichiometry (-)",
    "STC_cat": "Stoichiometry (-)",
    "XH2_dry": "Dry H2 mole fraction (-)",
    "FR_CL": "Coolant flow (NL/min)",
    "I": "Current (A)",
    "sa": "Anode saturation (-)",
    "ua": "Anode velocity (m/s)",
    "lambda_airin": "Air inlet stoichiometry (-)",
}


def lag_sequence(max_lag: int, mode: str = "sparse") -> list[int]:
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if mode == "dense":
        return list(range(0, max_lag + 1))
    if mode != "sparse":
        raise ValueError("mode must be 'sparse' or 'dense'")
    base = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]
    return [lag for lag in base if lag <= max_lag]


def prediction_mode_for_target(prediction_mode: str | dict[str, str], target: str) -> str:
    mode = prediction_mode if isinstance(prediction_mode, str) else prediction_mode[target]
    if mode not in {"absolute", "delta"}:
        raise ValueError("prediction_mode values must be 'absolute' or 'delta'")
    return mode


def _restore_standard_scaler(mean, scale) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    scaler.scale_ = np.asarray(scale, dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler


def _read_csv_skipping_unit_row(path: Path, required_inputs: list[str]) -> pd.DataFrame:
    preview = pd.read_csv(path, nrows=1, dtype=str)
    skiprows = None
    if not preview.empty and set(required_inputs).issubset(preview.columns):
        first_row = preview.iloc[0][required_inputs]
        numeric_values = pd.to_numeric(first_row, errors="coerce")
        t_is_unit = pd.isna(numeric_values["t"])
        mostly_non_numeric = numeric_values.isna().sum() >= max(1, len(required_inputs) // 2)
        if t_is_unit and mostly_non_numeric:
            skiprows = [1]
    return pd.read_csv(path, skiprows=skiprows)


def _lagged_frame_to_sequence_array(
    X: pd.DataFrame,
    lags: list[int],
    input_columns: list[str],
    feedback_output_columns: list[str],
) -> np.ndarray:
    ordered_lags = sorted(lags, reverse=True)
    sequence_columns = [
        [
            *(f"{column}_xlag{lag}" for column in input_columns),
            *(f"{column}_ylag{lag}" for column in feedback_output_columns),
        ]
        for lag in ordered_lags
    ]
    return np.stack([X[columns].to_numpy(dtype=np.float32) for columns in sequence_columns], axis=1)


class LoadedJointGRURegressor:
    def __init__(self, checkpoint_path: str | Path, input_columns: list[str]):
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        self.input_columns = list(input_columns)
        self.lags = list(checkpoint["lags"])
        self.params = dict(checkpoint["params"])
        self.output_columns = list(checkpoint["output_columns"])
        self.feedback_output_columns = list(checkpoint.get("feedback_output_columns", []))
        self.x_scaler = _restore_standard_scaler(checkpoint["x_scaler_mean"], checkpoint["x_scaler_scale"])
        self.y_scaler = _restore_standard_scaler(checkpoint["y_scaler_mean"], checkpoint["y_scaler_scale"])
        self.model = self._make_model(self.x_scaler.n_features_in_)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _make_model(self, n_features: int) -> nn.Module:
        class _JointGRUNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, output_columns, head_hidden_size):
                super().__init__()
                self.gru = nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.output_columns = list(output_columns)
                self.heads = nn.ModuleDict(
                    {
                        target: nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, 1),
                        )
                        for target in output_columns
                    }
                )

            def forward(self, x):
                out, _ = self.gru(x)
                last_hidden = out[:, -1, :]
                return torch.cat([self.heads[target](last_hidden) for target in self.output_columns], dim=1)

        return _JointGRUNet(
            input_size=n_features,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
            output_columns=self.output_columns,
            head_hidden_size=self.params.get("joint_head_hidden_size", self.params["hidden_size"]),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_seq = _lagged_frame_to_sequence_array(
            X,
            self.lags,
            self.input_columns,
            self.feedback_output_columns,
        )
        n_samples, n_steps, n_features = X_seq.shape
        X_scaled = self.x_scaler.transform(X_seq.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
        return self.y_scaler.inverse_transform(pred_scaled)


def read_trajectory(path: str | Path, trajectory_id: str, input_columns: list[str], output_columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    required_inputs = ["t", *input_columns]
    df = _read_csv_skipping_unit_row(path, required_inputs)
    missing = sorted(set(required_inputs) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required input columns: {missing}")
    for column in output_columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df[["t", *input_columns, *output_columns]].copy()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # if "STC_ano" in df.columns:
    #     df.loc[df["STC_ano"] == 0, "STC_ano"] = 1.6
    df = df.dropna(subset=required_inputs).copy()
    df["trajectory_id"] = trajectory_id
    df["source_file"] = path.name
    return df.sort_values("t").drop_duplicates("t", keep="first").reset_index(drop=True)


def resample_trajectory(
    group: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
    dt_seconds: float,
) -> pd.DataFrame:
    group = group.sort_values("t").drop_duplicates("t", keep="first").reset_index(drop=True)
    t_start = float(np.ceil(group["t"].min() / dt_seconds) * dt_seconds)
    t_end = float(np.floor(group["t"].max() / dt_seconds) * dt_seconds)
    t_new = np.arange(t_start, t_end + 0.5 * dt_seconds, dt_seconds)
    out = pd.DataFrame({"t": t_new})
    for column in [*input_columns, *output_columns]:
        values = group[column].to_numpy(dtype=float)
        out[column] = np.nan if np.isnan(values).all() else np.interp(t_new, group["t"].to_numpy(), values)
    out["trajectory_id"] = group["trajectory_id"].iloc[0]
    out["source_file"] = group["source_file"].iloc[0]
    return out


def _resolve_lags(lag, lag_mode: str) -> list[int]:
    if lag is None:
        return []
    if isinstance(lag, int):
        return lag_sequence(lag, lag_mode)
    return sorted(set(lag))


def _feature_row(
    inputs: pd.DataFrame,
    y_values: np.ndarray,
    index: int,
    input_columns: list[str],
    feedback_output_columns: list[str],
    x_lag,
    y_lag,
    lag_mode: str,
    feature_columns: list[str],
    pad_start_inputs: bool,
) -> pd.DataFrame:
    values = {}
    for lag in _resolve_lags(x_lag, lag_mode):
        source_index = index - lag
        if pad_start_inputs:
            source_index = max(source_index, 0)
        for column in input_columns:
            values[f"{column}_xlag{lag}"] = float(inputs.iloc[source_index][column])

    output_index = {column: i for i, column in enumerate(feedback_output_columns)}
    for lag in _resolve_lags(y_lag, lag_mode):
        for column in feedback_output_columns:
            values[f"{column}_ylag{lag}"] = float(y_values[index - lag, output_index[column]])
    return pd.DataFrame([[values[column] for column in feature_columns]], columns=feature_columns)


def _hard_physical_gate_mask(
    trajectory: pd.DataFrame,
    current_threshold: float = 0.0,
    stc_threshold: float = 0.0,
) -> pd.Series:
    inactive = pd.Series(False, index=trajectory.index)
    if "I" in trajectory.columns:
        inactive |= trajectory["I"].isna() | (trajectory["I"] <= current_threshold)
    for column in ("STC_ano", "STC_cat"):
        if column in trajectory.columns:
            inactive |= trajectory[column].isna() | (trajectory[column] <= stc_threshold)
    return inactive.fillna(True)


def free_run_predict_joint(
    trajectory: pd.DataFrame,
    model: LoadedJointGRURegressor,
    feature_columns: list[str],
    input_columns: list[str],
    output_columns: list[str],
    x_lag,
    y_lag,
    feedback_output_columns: list[str],
    lag_mode: str,
    stride: int,
    prediction_mode: str | dict[str, str],
    pad_start_inputs: bool = True,
    start_index: int = 0,
    eval_start_index: int = 1,
    clip_physical: bool = True,
    hard_physical_gate: bool = True,
    current_threshold: float = 0.0,
    stc_threshold: float = 0.0,
) -> pd.DataFrame:
    target_modes = {target: prediction_mode_for_target(prediction_mode, target) for target in output_columns}
    trajectory = trajectory.sort_values("t").reset_index(drop=True)
    x_lags = _resolve_lags(x_lag, lag_mode)
    y_lags = _resolve_lags(y_lag, lag_mode)
    max_x_lag = max(x_lags) if x_lags else 0
    max_y_lag = max(y_lags) if y_lags else 0
    min_start_index = max_y_lag if pad_start_inputs else max(max_x_lag, max_y_lag)
    start_index = max(start_index, min_start_index)
    if start_index >= len(trajectory) - 1:
        raise ValueError("start_index leaves no room for prediction")

    feedback_output_columns = list(feedback_output_columns)
    prediction_values = trajectory[output_columns].to_numpy(dtype=float).copy()
    y_values = trajectory[feedback_output_columns].to_numpy(dtype=float).copy() if feedback_output_columns else np.empty((len(trajectory), 0))
    output_index = {target: i for i, target in enumerate(output_columns)}
    feedback_index = {target: i for i, target in enumerate(feedback_output_columns)}
    forced_zero_mask = (
        _hard_physical_gate_mask(trajectory, current_threshold, stc_threshold).to_numpy(dtype=bool)
        if hard_physical_gate
        else np.zeros(len(trajectory), dtype=bool)
    )
    prediction_values[forced_zero_mask, :] = 0.0
    if feedback_output_columns:
        y_values[forced_zero_mask, :] = 0.0

    for index in range(start_index, len(trajectory) - 1, stride):
        end_index = min(index + stride, len(trajectory) - 1)
        target_slice = slice(index + 1, end_index + 1)
        if forced_zero_mask[target_slice].all():
            prediction_values[target_slice, :] = 0.0
            if feedback_output_columns:
                y_values[target_slice, :] = 0.0
            continue

        row = _feature_row(
            trajectory,
            y_values,
            index,
            input_columns,
            feedback_output_columns,
            x_lag,
            y_lag,
            lag_mode,
            feature_columns,
            pad_start_inputs,
        )
        model_outputs = np.asarray(model.predict(row)[0], dtype=float)
        next_values = {}
        for target_index, target in enumerate(output_columns):
            if target_modes[target] == "delta":
                current_value = float(y_values[index, feedback_index[target]]) if target in feedback_index else float(trajectory.iloc[index][target])
                next_value = current_value + float(model_outputs[target_index])
            else:
                next_value = float(model_outputs[target_index])
            if clip_physical and target in {"sa", "ua", "lambda_airin"}:
                next_value = max(next_value, 1e-9 if target == "ua" else 0.0)
            next_values[target] = next_value

        for target, next_value in next_values.items():
            prediction_values[index + 1 : end_index + 1, output_index[target]] = next_value
            if target in feedback_index:
                y_values[index + 1 : end_index + 1, feedback_index[target]] = next_value
        if forced_zero_mask[target_slice].any():
            prediction_values[target_slice, :] = np.where(
                forced_zero_mask[target_slice, None],
                0.0,
                prediction_values[target_slice, :],
            )
            if feedback_output_columns:
                y_values[target_slice, :] = np.where(
                    forced_zero_mask[target_slice, None],
                    0.0,
                    y_values[target_slice, :],
                )

    passthrough_columns = ["I"] if "I" in trajectory.columns else []
    out = trajectory[["trajectory_id", "source_file", "t", *passthrough_columns, *output_columns]].copy()
    for target in output_columns:
        out[f"{target}_pred"] = prediction_values[:, output_index[target]]
    out["is_warmup"] = np.arange(len(out)) < eval_start_index
    out["is_seed"] = np.arange(len(out)) <= start_index
    out["is_startup_padded"] = bool(pad_start_inputs) & (np.arange(len(out)) <= max_x_lag)
    out["is_forced_zero"] = forced_zero_mask
    return out


def regression_metrics(y_true, y_pred, label: str, target: str) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "label": label,
        "target": target,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "max_abs_percentage_error": float(np.max(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100)),
        "p95_abs_percentage_error": float(np.quantile(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100, 0.95)),
        "p99_abs_percentage_error": float(np.quantile(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100, 0.99)),
    }


def run_prediction(
    model_config_path: str | Path,
    model_path: str | Path,
    data_paths: dict[str, str | Path],
    output_dir: str | Path,
    show_plots: bool = False,
    hard_physical_gate: bool = True,
    current_threshold: float = 0.0,
    stc_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(model_config_path, "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    input_columns = list(saved_config["input_columns"])
    output_columns = list(saved_config["output_columns"])
    resample_dt_seconds = float(saved_config["resample_dt_seconds"])
    lag_mode = saved_config.get("lag_mode", "sparse")
    free_run_stride = int(saved_config.get("free_run_stride", 1))
    model_name = saved_config.get("default_model", "B_exogenous_u_only")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LoadedJointGRURegressor(model_path, input_columns)
    feature_columns = saved_config.get("feature_columns_by_model", {}).get(model_name)
    if feature_columns is None:
        feature_columns = [
            column
            for lag in sorted(model.lags, reverse=True)
            for column in [
                *(f"{name}_xlag{lag}" for name in input_columns),
                *(f"{name}_ylag{lag}" for name in model.feedback_output_columns),
            ]
        ]
    prediction_mode = saved_config.get(
        "exogenous_prediction_mode",
        {target: "absolute" for target in output_columns},
    )

    predictions = []
    metric_records = []
    checks = []
    for trajectory_id, path in data_paths.items():
        trajectory_output_dir = output_dir / trajectory_id
        trajectory_output_dir.mkdir(parents=True, exist_ok=True)

        raw = read_trajectory(path, trajectory_id, input_columns, output_columns)
        resampled = resample_trajectory(raw, input_columns, output_columns, resample_dt_seconds)
        if len(resampled) <= 2:
            print(f"Skip {trajectory_id}: too short for free-run prediction")
            continue

        pred = free_run_predict_joint(
            resampled,
            model=model,
            feature_columns=feature_columns,
            input_columns=input_columns,
            output_columns=output_columns,
            x_lag=model.lags,
            y_lag=model.lags if model.feedback_output_columns else [],
            feedback_output_columns=model.feedback_output_columns,
            lag_mode=lag_mode,
            stride=free_run_stride,
            prediction_mode=prediction_mode,
            pad_start_inputs=True,
            start_index=0,
            eval_start_index=1,
            hard_physical_gate=hard_physical_gate,
            current_threshold=current_threshold,
            stc_threshold=stc_threshold,
        )
        pred["model"] = model_name
        pred["split"] = "external"
        predictions.append(pred)
        pred.to_csv(trajectory_output_dir / f"{trajectory_id}_free_run_predictions.csv", index=False)

        trajectory_metric_records = []
        eval_mask = ~pred["is_seed"]
        for target in output_columns:
            target_mask = eval_mask & pred[target].notna()
            if target_mask.any():
                record = regression_metrics(
                    pred.loc[target_mask, target],
                    pred.loc[target_mask, f"{target}_pred"],
                    f"external:{trajectory_id}:full_from_t0",
                    target,
                )
                record["model"] = model_name
                record["trajectory_id"] = trajectory_id
                record["evaluation_scope"] = "full_from_t0"
                metric_records.append(record)
                trajectory_metric_records.append(record)
            lower_bound = 0.0
            check = {
                "trajectory_id": trajectory_id,
                "model": model_name,
                "check": f"{target}_physical_lower_bound",
                "passed": bool((pred[f"{target}_pred"] >= lower_bound).all()),
                "min_value": float(pred[f"{target}_pred"].min()),
                "forced_zero_rows": int(pred["is_forced_zero"].sum()),
            }
            checks.append(check)

        if trajectory_metric_records:
            pd.DataFrame(trajectory_metric_records).to_csv(
                trajectory_output_dir / f"{trajectory_id}_free_run_metrics.csv",
                index=False,
            )
        _plot_prediction(pred, output_columns, trajectory_output_dir, model_name, show_plots)
        _plot_prediction_with_current(pred, resampled, output_columns, trajectory_output_dir, model_name, show_plots)
        _plot_input_trajectories(resampled, input_columns, trajectory_output_dir, show_plots)

    if not predictions:
        raise ValueError("No predictions were generated.")

    prediction_df = pd.concat(predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metric_records).sort_values(["target", "trajectory_id"]).reset_index(drop=True) if metric_records else pd.DataFrame()
    checks_df = pd.DataFrame(checks)
    return prediction_df, metrics_df, checks_df


def _plot_prediction(
    prediction: pd.DataFrame,
    output_columns: list[str],
    output_dir: Path,
    model_name: str,
    show_plots: bool,
) -> None:
    trajectory_id = prediction["trajectory_id"].iloc[0]
    plt.rcParams["figure.figsize"] = (12, 4)
    plt.rcParams["axes.grid"] = True
    fig, axes = plt.subplots(len(output_columns), 1, figsize=(12, 9), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, output_columns):
        if prediction[target].notna().any():
            ax.plot(prediction["t"], prediction[target], label=f"{target} true", color="black", linewidth=1.8)
        ax.plot(
            prediction["t"],
            prediction[f"{target}_pred"],
            label=f"{target}_pred",
            color="tab:orange",
            linewidth=1.2,
            alpha=0.9,
        )
        startup = prediction[prediction["is_startup_padded"]]
        if not startup.empty:
            ax.axvspan(
                startup["t"].min(),
                startup["t"].max(),
                color="gray",
                alpha=0.12,
                label="startup padded" if target == output_columns[0] else None,
            )
        ax.set_ylabel(target)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"free-run prediction: {trajectory_id}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{trajectory_id}_free_run_prediction.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_prediction_with_current(
    prediction: pd.DataFrame,
    trajectory: pd.DataFrame,
    output_columns: list[str],
    output_dir: Path,
    model_name: str,
    show_plots: bool,
) -> None:
    trajectory_id = prediction["trajectory_id"].iloc[0]
    panel_columns = [*output_columns, "I"] if "I" in trajectory.columns else list(output_columns)
    fig, axes = plt.subplots(len(panel_columns), 1, figsize=(12, 3.0 * len(panel_columns)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, target in zip(axes, panel_columns):
        if target == "I":
            ax.plot(trajectory["t"], trajectory[target], label="I", color="tab:blue", linewidth=1.4)
        else:
            if prediction[target].notna().any():
                ax.plot(prediction["t"], prediction[target], label=f"{target} true", color="black", linewidth=1.8)
            ax.plot(
                prediction["t"],
                prediction[f"{target}_pred"],
                label=f"{target}_pred",
                color="tab:orange",
                linewidth=1.2,
                alpha=0.9,
            )
            startup = prediction[prediction["is_startup_padded"]]
            if not startup.empty:
                ax.axvspan(
                    startup["t"].min(),
                    startup["t"].max(),
                    color="gray",
                    alpha=0.12,
                    label="startup padded" if target == output_columns[0] else None,
                )
        ax.set_title(target)
        ax.set_ylabel(PLOT_YLABELS.get(target, target))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"free-run prediction with current: {trajectory_id}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / f"{trajectory_id}_free_run_prediction_with_current.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_input_trajectories(
    trajectory: pd.DataFrame,
    input_columns: list[str],
    output_dir: Path,
    show_plots: bool,
) -> None:
    trajectory_id = trajectory["trajectory_id"].iloc[0]
    plot_columns = [column for column in input_columns if column in trajectory.columns]
    if not plot_columns:
        return

    n_cols = 2
    n_rows = (len(plot_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.0 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, plot_columns):
        ax.plot(trajectory["t"], trajectory[column], linewidth=1.5)
        ax.set_title(column)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(PLOT_YLABELS.get(column, column))
        ax.grid(True, alpha=0.3)

    for ax in axes[len(plot_columns):]:
        ax.axis("off")

    fig.suptitle(f"input trajectories: {trajectory_id}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.985], w_pad=3.0, h_pad=1.6)
    fig.savefig(output_dir / f"{trajectory_id}_input_trajectories.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
