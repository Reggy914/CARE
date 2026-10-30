"""Training, evaluation, and artifact controller for the ELE dataset."""
from __future__ import annotations

import argparse
import copy
import importlib
from functools import lru_cache
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT_CONFIG_PATH = ROOT / "configs" / "experiments" / "default.yaml"

from utils.evaluator import build_evaluation_region_masks, compute_diagnostic_metrics, compute_metrics
from data_provider.ele import DEFAULT_ELE_DATA_DIR, ELE_DATASET_ID, build_ele_loaders, ele_target_columns
from utils.predictor import save_predictions

MODEL_CONFIG_DIR = ROOT / "configs" / "models"
GLOBAL_TRAINING_KEYS = frozenset({"input_len", "pred_len", "epochs", "batch_size", "learning_rate", "weight_decay", "grad_clip", "stride", "early_stopping", "early_stopping_patience", "early_stopping_min_delta"})
MODEL_METADATA_KEYS = frozenset({"module", "class_name", "inherits"})
DIAGNOSTIC_METRICS = [
    "P90_AE", "P95_AE", "P99_AE", "Max_AE",
    "Tail_MAE", "Tail_MSE", "Tail_RMSE",
    "HighAmplitude_MAE", "HighAmplitude_MSE", "HighAmplitude_RMSE",
    "Boundary_MAE", "Boundary_MSE", "Boundary_RMSE",
]
PER_SEED_RANKS = {
    "MAE": "rank_by_mae_in_target",
    "MSE": "rank_by_mse_in_target",
    "Tail_MAE": "rank_by_tail_mae_in_target",
    "Boundary_RMSE": "rank_by_boundary_rmse_in_target",
    "P95_AE": "rank_by_p95_ae_in_target",
    "P99_AE": "rank_by_p99_ae_in_target",
}

AGGREGATE_RANKS = {
    "MAE": "rank_by_MAE_mean",
    "MSE": "rank_by_MSE_mean",
    "Tail_MAE": "rank_by_Tail_MAE_mean",
    "Boundary_RMSE": "rank_by_Boundary_RMSE_mean",
    "P95_AE": "rank_by_P95_AE_mean",
    "P99_AE": "rank_by_P99_AE_mean",
}
METRIC_COLUMNS = [
    "rank_by_mae_global", *PER_SEED_RANKS.values(),
    "dataset", "target_column", "target_index", "seed", "model_order", "model", "teacher_model", "status",
    "MAE", "MSE", "RMSE", "MAPE", "R2", *DIAGNOSTIC_METRICS,
    "train_time_sec", "predict_time_sec", "total_time_sec", "artifact_dir", "error_message", "notes",
]
SUMMARY_METRICS = [
    "MAE", "MSE", "RMSE", "MAPE", "R2", *DIAGNOSTIC_METRICS,
    "train_time_sec", "predict_time_sec", "total_time_sec",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-") or "item"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dataset_directory(dataset: str) -> Path:
    """Canonical storage path: nested dataset identifiers stay nested."""
    parts = [part for part in str(dataset).replace(chr(92), "/").split("/") if part]
    return Path(*[safe(part) for part in parts]) if parts else Path("unknown_dataset")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_log(run_dir: Path, message: str) -> None:
    with (run_dir / "main.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")


@lru_cache(maxsize=1)
def model_config_registry() -> dict[str, dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}
    paths = sorted(MODEL_CONFIG_DIR.glob("*.yaml"), key=lambda item: item.name)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Model config must be a YAML mapping: {path}")
        model_id = path.stem
        duplicated_global_keys = sorted(GLOBAL_TRAINING_KEYS & set(payload))
        if duplicated_global_keys:
            raise ValueError(f"{path}: global parameters belong only in {EXPERIMENT_CONFIG_PATH}: {duplicated_global_keys}")
        if model_id in raw:
            raise ValueError(f"Duplicate model config id {model_id!r}: {path}")
        raw[model_id] = payload

    def resolve(model_id: str, ancestry: tuple[str, ...] = ()) -> dict[str, Any]:
        if model_id not in raw:
            raise ValueError(f"Unknown model id in inheritance chain: {model_id!r}")
        if model_id in ancestry:
            raise ValueError(f"Cyclic model inheritance: {(*ancestry, model_id)}")
        payload = raw[model_id]
        parent = payload.get("inherits")
        merged = resolve(str(parent), (*ancestry, model_id)) if parent else {}
        merged.update({key: copy.deepcopy(value) for key, value in payload.items() if key != "inherits"})
        return merged

    return {model_id: resolve(model_id) for model_id in raw}


def model_config_for(model_id: str) -> dict[str, Any]:
    registry = model_config_registry()
    if model_id not in registry:
        raise ValueError(f"unknown model {model_id!r}; available: {list(registry)}")
    return copy.deepcopy(registry[model_id])


def model_runtime_config(model_id: str) -> dict[str, Any]:
    config = model_config_for(model_id)
    return {key: value for key, value in config.items() if key not in MODEL_METADATA_KEYS}


def apply_external_model_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply explicit local checkpoint paths without mutating model YAML files."""
    updated = copy.deepcopy(config)
    timer_model_dir = getattr(args, "timer_model_dir", None)
    if timer_model_dir is not None:
        updated["timer_model_dir"] = str(Path(timer_model_dir).resolve())
    return updated


def model_class(model_id: str) -> type:
    config = model_config_for(model_id)
    module_name, class_name = config.get("module"), config.get("class_name")
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        raise ValueError(f"{model_id}: model YAML must define module and class_name")
    return getattr(importlib.import_module(module_name), class_name)


def concatenate_truth(loader: Any) -> tuple[np.ndarray, np.ndarray]:
    truth, starts = [], []
    for batch in loader:
        truth.append(batch[1].detach().cpu().numpy()); starts.append(batch[2].detach().cpu().numpy())
    return np.concatenate(truth), np.concatenate(starts)


def concatenate_evaluation_data(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth, starts, last_observed = [], [], []
    for batch in loader:
        x = batch[0].detach().cpu().numpy()
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"evaluation expects univariate [B, L, 1] inputs, got {x.shape}")
        truth.append(batch[1].detach().cpu().numpy())
        starts.append(batch[2].detach().cpu().numpy())
        last_observed.append(x[:, -1, 0])
    return np.concatenate(truth), np.concatenate(starts), np.concatenate(last_observed)


def inverse(scaler: Any, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape).astype(np.float32)


def standard_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for field in METRIC_COLUMNS:
        if field not in frame: frame[field] = pd.NA
    group_columns = ["dataset", "target_column", "seed"]
    for metric, rank_field in PER_SEED_RANKS.items():
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        eligible = frame["status"].eq("SUCCESS") & numeric.notna()
        candidates = frame.loc[eligible, group_columns].copy()
        candidates["_metric"] = numeric.loc[eligible]
        ranked = candidates.groupby(group_columns, dropna=False)["_metric"].rank(method="min", ascending=True)
        rank_values = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        rank_values.loc[ranked.index] = ranked.astype("Int64")
        frame[rank_field] = rank_values
    return frame[METRIC_COLUMNS].sort_values("model_order", kind="stable")


def save_outputs(rows: list[dict[str, Any]], target: str, run_dir: Path, result_dir: Path) -> None:
    target_artifact = run_dir / f"target_{safe(target)}"
    tables = result_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame = standard_rows(rows)
    frame.to_csv(tables / "metrics.csv", index=False)
    try: frame.to_markdown(tables / "metrics.md", index=False)
    except Exception: (tables / "metrics.md").write_text(frame.to_csv(index=False), encoding="utf-8")
    write_json(target_artifact / "metrics" / "metrics.json", {"target_column": target, "rows": frame.to_dict("records")})


def model_rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize five-seed per-target ranks without mixing target scales."""
    records: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame()
    for keys, group in summary.groupby(["dataset", "model_order", "model"], dropna=False, sort=False):
        dataset, model_order, model = keys
        complete = group[group["status"].eq("SUCCESS")]
        row: dict[str, Any] = {
            "dataset": dataset,
            "model_order": model_order,
            "model": model,
            "target_count": int(group["target_column"].nunique()),
            "complete_target_count": int(complete["target_column"].nunique()),
        }
        for metric, rank_field in AGGREGATE_RANKS.items():
            metric_values = pd.to_numeric(complete[f"{metric}_mean"], errors="coerce").dropna()
            ranks = pd.to_numeric(complete[rank_field], errors="coerce").dropna()
            row[f"{metric}_mean_across_targets"] = float(metric_values.mean()) if len(metric_values) else np.nan
            row[f"{metric}_average_rank"] = float(ranks.mean()) if len(ranks) else np.nan
            row[f"{metric}_rank1_target_count"] = int(ranks.eq(1).sum())
            row[f"{metric}_ranked_target_count"] = int(len(ranks))
        records.append(row)
    return pd.DataFrame(records).sort_values("model_order", kind="stable")


def aggregate_seed_results(rows: list[dict[str, Any]], parent_run_dir: Path, dataset: str, targets: list[str]) -> pd.DataFrame:
    per_seed = standard_rows(rows)
    tables_root = parent_run_dir / "aggregate" / "tables"; tables_root.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(tables_root / "per_seed_metrics.csv", index=False)
    successful = per_seed[per_seed["status"].eq("SUCCESS")].copy()
    group_keys = ["dataset", "target_column", "target_index", "model_order", "model"]
    records: list[dict[str, Any]] = []
    for keys, group in successful.groupby(group_keys, dropna=False, sort=False):
        row = dict(zip(group_keys, keys)); row["seed_count"] = int(group["seed"].nunique()); row["seeds"] = ",".join(str(int(x)) for x in sorted(group["seed"].astype(int).unique()))
        teacher_values = sorted({str(x) for x in group["teacher_model"].dropna() if str(x).strip()})
        row["teacher_model"] = ",".join(teacher_values)
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row["status"] = "SUCCESS" if row["seed_count"] == len(set(per_seed["seed"].dropna().astype(int))) else "PARTIAL"
        records.append(row)
    summary = pd.DataFrame(records)
    if not summary.empty:
        for metric, rank_field in AGGREGATE_RANKS.items():
            metric_column = f"{metric}_mean"
            numeric = pd.to_numeric(summary[metric_column], errors="coerce")
            eligible = summary["status"].eq("SUCCESS") & numeric.notna()
            candidates = summary.loc[eligible, ["target_column"]].copy()
            candidates["_metric"] = numeric.loc[eligible]
            ranked = candidates.groupby("target_column", dropna=False)["_metric"].rank(method="min", ascending=True)
            rank_values = pd.Series(pd.NA, index=summary.index, dtype="Int64")
            rank_values.loc[ranked.index] = ranked.astype("Int64")
            summary[rank_field] = rank_values
        summary = summary.sort_values(["target_index", "model_order"], kind="stable")
    summary.to_csv(tables_root / "summary_mean_std.csv", index=False)
    try: summary.to_markdown(tables_root / "summary_mean_std.md", index=False)
    except Exception: (tables_root / "summary_mean_std.md").write_text(summary.to_csv(index=False), encoding="utf-8")
    rank_summary = model_rank_summary(summary)
    rank_summary.to_csv(tables_root / "model_rank_summary.csv", index=False)
    try: rank_summary.to_markdown(tables_root / "model_rank_summary.md", index=False)
    except Exception: (tables_root / "model_rank_summary.md").write_text(rank_summary.to_csv(index=False), encoding="utf-8")

    for target in targets:
        target_summary = summary[summary["target_column"].eq(target)].copy() if not summary.empty else summary
        target_root = parent_run_dir / "aggregate" / f"target_{safe(target)}"; tables = target_root / "tables"
        tables.mkdir(parents=True, exist_ok=True); target_summary.to_csv(tables / "summary_mean_std.csv", index=False)
        if target_summary.empty: continue
        truth_reference: np.ndarray | None = None; id_reference: np.ndarray | None = None
        target_rows = successful[successful["target_column"].eq(target)]
        for model_id, group in target_rows.groupby("model", sort=False):
            seed_predictions=[]
            for item in group.sort_values("seed").to_dict("records"):
                pred_dir=Path(item["artifact_dir"])/"predictions"; truth=np.load(pred_dir/"y_true.npy"); pred=np.load(pred_dir/"y_pred.npy"); ids=np.load(pred_dir/"sample_id.npy")
                if truth_reference is None: truth_reference, id_reference = truth, ids
                elif not np.array_equal(ids,id_reference) or not np.array_equal(truth,truth_reference): raise RuntimeError(f"seed prediction alignment mismatch target={target} model={model_id}")
                seed_predictions.append(pred)
            averaged=np.mean(np.stack(seed_predictions,axis=0),axis=0)
            output=target_root/"predictions"/safe(str(model_id)); output.mkdir(parents=True,exist_ok=True); np.save(output/"y_pred_mean.npy",averaged); np.save(output/"y_true.npy",truth_reference); np.save(output/"sample_id.npy",id_reference)
    return summary


def run_target(args: argparse.Namespace, dataset: str, directory: Path, target: str, target_index: int, models: list[str], run_dir: Path) -> list[dict[str, Any]]:
    print(f"[{target_index}] {target}: loading data windows...", flush=True)
    bundle = build_ele_loaders(directory, target_column=target, input_len=args.input_len, pred_len=args.pred_len, stride=args.stride, batch_size=args.batch_size, max_train_windows=args.max_train_windows, max_val_windows=args.max_val_windows, max_test_windows=args.max_test_windows)
    print(f"[{target_index}] {target}: data ready; starting {len(models)} models", flush=True)
    write_json(run_dir / f"target_{safe(target)}" / "split_info.json", bundle.split_info)
    truth_scaled, sample_ids, last_observed_scaled = concatenate_evaluation_data(bundle.test_loader)
    truth = inverse(bundle.scaler, truth_scaled)
    last_observed = inverse(bundle.scaler, last_observed_scaled)
    region_statistics = copy.deepcopy(bundle.split_info["evaluation_region_statistics"])
    region_masks = build_evaluation_region_masks(truth, last_observed, region_statistics)
    region_statistics["test_regions"] = {
        name: {
            "point_count": int(mask.sum()),
            "point_fraction": float(mask.mean()),
        }
        for name, mask in region_masks.items()
    }
    region_statistics["test_total_points"] = int(region_masks["tail"].size)
    target_metrics_dir = run_dir / f"target_{safe(target)}" / "metrics"
    write_json(target_metrics_dir / "evaluation_region_statistics.json", region_statistics)
    np.savez_compressed(
        target_metrics_dir / "evaluation_region_masks.npz",
        sample_id=sample_ids,
        tail=region_masks["tail"],
        boundary=region_masks["boundary"],
        high_amplitude=region_masks["high_amplitude"],
    )
    val_truth_scaled, _ = concatenate_truth(bundle.val_loader)
    # Distill-LSTM is intentionally run after the four candidate teachers. Keep
    # their validation scores and checkpoints so selection never uses test data.
    from exp.teacher_selection import CANDIDATE_MODELS, run_teacher_selection
    teacher_candidate_dirs: dict[str, Path] = {}
    teacher_candidate_status: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for order, model_id in enumerate(models, 1):
        artifact = run_dir / f"target_{safe(target)}" / model_id
        log_dir = artifact / "logs"; started = time.time()
        status_path = log_dir / "run_status.json"
        required_outputs = [
            artifact / "metrics" / "test_metrics.json",
            artifact / "predictions" / "y_true.npy",
            artifact / "predictions" / "y_pred.npy",
            artifact / "predictions" / "sample_id.npy",
        ]
        if model_id in CANDIDATE_MODELS:
            required_outputs.append(artifact / "checkpoints" / "best_model.pt")
        if getattr(args, "resume", False) and status_path.exists() and all(path.exists() for path in required_outputs):
            try:
                resumed_row = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                resumed_row = {}
            if resumed_row.get("status") == "SUCCESS":
                row = resumed_row
                if model_id in CANDIDATE_MODELS:
                    teacher_candidate_dirs[model_id] = artifact
                    teacher_candidate_status[model_id] = {
                        "status": row["status"],
                        "metrics": row.get("validation_metrics", {}),
                        "train_time_sec": row["train_time_sec"],
                        "error_message": row.get("error_message", ""),
                    }
                append_log(run_dir, f"MODEL_RESUME_SKIP target={target} model={model_id} status=SUCCESS")
                print(f"[{target_index}:{order}/{len(models)}] {model_id} {target}: RESUME_SKIP (verified SUCCESS artifacts)", flush=True)
                rows.append(row)
                continue
        try:
            set_global_seed(int(args.seed))
            # Model YAML supplies only model-specific settings. default.yaml and
            # explicit CLI arguments always win for global training parameters.
            config = {**apply_external_model_paths(model_runtime_config(model_id), args), **dict(args._effective_config),
                      "device": args.device, "dataset": dataset, "target_column": target,
                      "processed_path": str(directory / "ele_processed.csv"), "input_len": args.input_len, "pred_len": args.pred_len,
                      "epochs": args.epochs, "batch_size": args.batch_size, "stride": args.stride, "learning_rate": args.learning_rate,
                      "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "seed": args.seed}
            if model_id == "distill":
                config["distill_cache_dir"] = str(args.output_dir.expanduser().resolve() / "cache" / "distill")
            teacher_model = ""
            if model_id == "distill" and not config.get("teacher_dir"):
                missing_candidates = [name for name in CANDIDATE_MODELS if name not in teacher_candidate_status]
                if missing_candidates:
                    raise RuntimeError(
                        "Distill-LSTM requires its four candidate teachers to run first in the same "
                        f"experiment, or an explicit teacher_dir. Missing: {missing_candidates}"
                    )
                selection = run_teacher_selection(
                    dataset_id=dataset,
                    dataset_path=str(directory),
                    target_column=target,
                    input_len=args.input_len,
                    pred_len=args.pred_len,
                    stride=args.stride,
                    seed=args.seed,
                    scaler_info={
                        "mean": np.asarray(getattr(bundle.scaler, "mean_", []), dtype=float).tolist(),
                        "scale": np.asarray(getattr(bundle.scaler, "scale_", []), dtype=float).tolist(),
                    },
                    split_info=bundle.split_info,
                    model_checkpoint_dirs=teacher_candidate_dirs,
                    model_run_status=teacher_candidate_status,
                )
                config["teacher_dir"] = selection["teacher_dir"]
                teacher_model = str(selection["selection"].get("selected_model", ""))
            write_json(log_dir / "effective_model_config.json", config)
            append_log(run_dir, f"MODEL_START target={target} model={model_id}")
            print(f"[{target_index}:{order}/{len(models)}] {model_id} {target}: loading model", flush=True)
            model = model_class(model_id)(config)
            print(f"[{target_index}:{order}/{len(models)}] {model_id} {target}: training", flush=True)
            train_started = time.time(); fit_result = model.fit(bundle.train_loader, bundle.val_loader); train_sec = time.time() - train_started
            write_json(log_dir / "fit_result.json", fit_result)
            if isinstance(fit_result, dict) and isinstance(fit_result.get("history"), list):
                pd.DataFrame(fit_result["history"]).to_csv(log_dir / "training_history.csv", index=False)
            if isinstance(fit_result, dict) and isinstance(fit_result.get("training_control"), dict):
                write_json(log_dir / "best_epoch.json", fit_result["training_control"])
            print(f"[{target_index}:{order}/{len(models)}] {model_id} {target}: training done in {train_sec:.1f}s; predicting", flush=True)
            pred_started = time.time(); predicted_scaled = np.asarray(model.predict(bundle.test_loader), dtype=np.float32); pred_sec = time.time() - pred_started
            print(f"[{target_index}:{order}/{len(models)}] {model_id} {target}: prediction done in {pred_sec:.1f}s; saving", flush=True)
            predicted = inverse(bundle.scaler, predicted_scaled)
            metrics = {
                **compute_metrics(truth, predicted),
                **compute_diagnostic_metrics(truth, predicted, region_masks),
            }
            write_json(artifact / "metrics" / "test_metrics.json", metrics)
            model_diagnostics = getattr(model, "last_prediction_diagnostics", None)
            if isinstance(model_diagnostics, dict):
                write_json(artifact / "metrics" / "model_diagnostics.json", model_diagnostics)
            validation_metrics: dict[str, float] | None = None
            if model_id in CANDIDATE_MODELS:
                val_predicted_scaled = np.asarray(model.predict(bundle.val_loader), dtype=np.float32)
                val_predicted = inverse(bundle.scaler, val_predicted_scaled)
                val_truth = inverse(bundle.scaler, val_truth_scaled)
                validation_metrics = compute_metrics(val_truth, val_predicted)
            save_predictions(artifact / "predictions", truth, predicted, sample_ids)
            if hasattr(model, "save"): model.save(artifact / "checkpoints" / "best_model.pt")
            row = {"dataset": dataset, "target_column": target, "target_index": target_index, "model_order": order, "model": model_id, "teacher_model": teacher_model, "status": "SUCCESS", **metrics,
                   "seed": int(args.seed),
                   "train_time_sec": train_sec, "predict_time_sec": pred_sec, "total_time_sec": time.time()-started, "artifact_dir": str(artifact), "error_message": "", "notes": ""}
            if validation_metrics is not None:
                row["validation_metrics"] = validation_metrics
        except Exception as exc:
            row = {"dataset": dataset, "target_column": target, "target_index": target_index, "model_order": order, "model": model_id, "teacher_model": "", "status": "FAILED",
                   "seed": int(args.seed),
                   "train_time_sec": 0.0, "predict_time_sec": 0.0, "total_time_sec": time.time()-started, "artifact_dir": str(artifact), "error_message": f"{type(exc).__name__}: {exc}", "notes": ""}
        if model_id in CANDIDATE_MODELS:
            teacher_candidate_dirs[model_id] = artifact
            teacher_candidate_status[model_id] = {
                "status": row["status"],
                "metrics": row.get("validation_metrics", {}),
                "train_time_sec": row["train_time_sec"],
                "error_message": row["error_message"],
            }
        write_json(log_dir / "run_status.json", row); (log_dir / "train.log").parent.mkdir(parents=True, exist_ok=True); (log_dir / "train.log").write_text(f"{row['status']} {row.get('error_message', '')}\n", encoding="utf-8")
        append_log(run_dir, f"MODEL_DONE target={target} model={model_id} status={row['status']} seconds={row['total_time_sec']:.3f}")
        print(f"[{target_index}:{order}/{len(models)}] {model_id} {target}: {row['status']}", flush=True); rows.append(row)
    result_dir = run_dir / f"target_{safe(target)}"
    save_outputs(rows, target, run_dir, result_dir)
    return rows


def load_experiment_config() -> dict[str, Any]:
    """Load the project's single ELE experiment configuration."""
    if not EXPERIMENT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Experiment config not found: {EXPERIMENT_CONFIG_PATH}")
    with EXPERIMENT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {EXPERIMENT_CONFIG_PATH}")
    return data


def parse_args() -> argparse.Namespace:
    experiment_cfg = load_experiment_config()
    required_keys = (
        "input_len", "pred_len", "stride", "epochs", "batch_size", "learning_rate",
        "weight_decay", "grad_clip", "early_stopping", "early_stopping_patience",
        "early_stopping_min_delta", "seeds", "models",
    )
    missing_keys = [key for key in required_keys if key not in experiment_cfg]
    if missing_keys:
        raise ValueError(f"{EXPERIMENT_CONFIG_PATH} is missing required keys: {missing_keys}")
    unexpected_keys = sorted(set(experiment_cfg) - set(required_keys))
    if unexpected_keys:
        raise ValueError(f"{EXPERIMENT_CONFIG_PATH} contains unused keys: {unexpected_keys}")

    parser = argparse.ArgumentParser(description="ELE forecasting controller")
    parser.add_argument("--ele-data-dir", type=Path, default=DEFAULT_ELE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Directory for all checkpoints, logs, metrics, and predictions")
    parser.add_argument("--timer-model-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device, for example cpu, cuda, or cuda:0")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target-columns", nargs="+")
    target_group.add_argument("--all-columns", action="store_true")
    parser.add_argument("--models", nargs="+", default=experiment_cfg["models"])
    parser.add_argument("--input-len", type=int, default=experiment_cfg["input_len"])
    parser.add_argument("--pred-len", type=int, default=experiment_cfg["pred_len"])
    parser.add_argument("--stride", type=int, default=experiment_cfg["stride"])
    parser.add_argument("--epochs", type=int, default=experiment_cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=experiment_cfg["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=experiment_cfg["learning_rate"])
    parser.add_argument("--seed", type=int, default=None, help="Single-seed override")
    parser.add_argument("--seeds", type=int, nargs="+", default=experiment_cfg["seeds"], help="Independent experiment seeds")
    parser.add_argument("--weight-decay", type=float, default=experiment_cfg["weight_decay"])
    parser.add_argument("--grad-clip", type=float, default=experiment_cfg["grad_clip"])
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-val-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--run-id", help="Stable artifact directory name for resumable jobs")
    parser.add_argument("--resume", action="store_true", help="Skip verified successful model artifacts")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    args._experiment_config_path = str(EXPERIMENT_CONFIG_PATH)
    args._effective_config = experiment_cfg
    return args

def main() -> int:
    args = parse_args()
    dataset = ELE_DATASET_ID
    directory = args.ele_data_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    os.environ.setdefault("CARE_PRETRAINED_ROOT", str(output_root / "pretrained"))
    available_targets = ele_target_columns(directory)
    targets = available_targets if args.all_columns else (args.target_columns or [available_targets[0]])
    bad_targets = [item for item in targets if item not in available_targets]
    if bad_targets: raise ValueError(f"unknown targets {bad_targets}; available: {available_targets}")
    available_models = model_config_registry()
    models = list(available_models) if args.models == ["all"] else args.models
    bad_models = [item for item in models if item not in available_models]
    if bad_models: raise ValueError(f"unknown models {bad_models}; available: {list(available_models)}")
    if args.dry_run:
        bundle = build_ele_loaders(directory, target_column=targets[0], input_len=args.input_len, pred_len=args.pred_len, stride=args.stride, batch_size=args.batch_size, max_train_windows=args.max_train_windows or 8, max_val_windows=args.max_val_windows or 8, max_test_windows=args.max_test_windows or 8)
        seeds=[int(args.seed)] if args.seed is not None else [int(x) for x in args.seeds]
        print(json.dumps({"dataset": dataset, "targets": targets, "models": models, "seeds": seeds, "loader": bundle.split_info}, ensure_ascii=False, indent=2)); return 0
    run_id = safe(args.run_id) if args.run_id else f"{safe(dataset)}_{utc_stamp()}"
    run_dir = output_root / dataset_directory(dataset) / run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists; pass --resume to continue: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    seeds=[int(args.seed)] if args.seed is not None else [int(x) for x in args.seeds]
    if not seeds or len(seeds)!=len(set(seeds)): raise ValueError(f"seeds must be a non-empty unique list, got {seeds}")
    run_config_record = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    run_config_record["dataset"] = dataset
    run_config_record["experiment_config_file"] = args._experiment_config_path
    existing_config_path = run_dir / "run_config.json"
    if args.resume and existing_config_path.exists():
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        identity_fields = (
            "dataset", "target_columns", "models", "input_len", "pred_len", "stride",
            "epochs", "batch_size", "learning_rate", "weight_decay", "grad_clip",
            "seed", "seeds", "max_train_windows", "max_val_windows", "max_test_windows",
        )
        mismatches = {
            key: (existing_config.get(key), run_config_record.get(key))
            for key in identity_fields
            if existing_config.get(key) != run_config_record.get(key)
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch for {run_dir}: {mismatches}")
    write_json(run_dir / "run_config.json", run_config_record)
    write_json(run_dir / "effective_config.json", args._effective_config)
    write_json(run_dir / "target_columns.json", {"current_cols": targets, "skipped_cols": []})
    append_log(run_dir, f"START dataset={dataset} targets={targets} models={models} seeds={seeds}")
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_args=copy.copy(args); seed_args.seed=int(seed); seed_dir=run_dir/f"seed_{seed}"; seed_dir.mkdir(parents=True,exist_ok=args.resume)
        write_json(seed_dir/"seed_manifest.json",{"seed":seed,"dataset":dataset,"targets":targets,"models":models})
        append_log(run_dir,f"SEED_START seed={seed}")
        seed_rows=[]
        for index,target in enumerate(targets,1): seed_rows.extend(run_target(seed_args,dataset,directory,target,index,models,seed_dir))
        all_rows.extend(seed_rows); seed_frame=standard_rows(seed_rows); seed_success=int(seed_frame["status"].eq("SUCCESS").sum())
        write_json(seed_dir/"run_summary.json",{"dataset":dataset,"parent_run_id":run_id,"seed":seed,"success_count":seed_success,"failed_count":len(seed_frame)-seed_success,"rows":seed_frame.to_dict("records")})
        append_log(run_dir,f"SEED_DONE seed={seed} success={seed_success} failed={len(seed_frame)-seed_success}")
    frame=standard_rows(all_rows); successful=int(frame["status"].eq("SUCCESS").sum()); summary=aggregate_seed_results(all_rows,run_dir,dataset,targets)
    write_json(run_dir / "run_summary.json", {"dataset": dataset, "run_id": run_id, "input_len": args.input_len, "pred_len": args.pred_len, "stride": args.stride, "targets": targets, "models": models, "seeds":seeds,"seed_count":len(seeds), "success_count": successful, "failed_count": len(frame)-successful, "per_seed_rows": frame.to_dict("records"),"aggregate_rows":summary.to_dict("records")})
    append_log(run_dir, f"FINISHED success={successful} failed={len(frame)-successful}"); print(f"completed: {run_dir}", flush=True); return 1 if len(frame) - successful else 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc: print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
