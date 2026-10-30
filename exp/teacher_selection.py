from __future__ import annotations

"""Teacher selection pipeline for the 4 candidate baseline models.

Candidates (fixed order for tie-breaking):
  1. PatchTST
  2. iTransformer
  3. TimeMixer
  4. GTR

Selection is performed per (dataset, target_column, input_len, pred_len,
stride, split_hash, seed, scaler_identity).  Only validation-set metrics are
used; test-set metrics MUST NOT influence selection.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
def pretrained_root_from_environment(environment: dict[str, str] | os._Environ[str]) -> Path:
    configured = environment.get("CARE_PRETRAINED_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else PROJECT_ROOT / "pretrained"
    )


PRETRAINED_ROOT = pretrained_root_from_environment(os.environ)

# Fixed candidate order (also serves as deterministic tie-breaker).
CANDIDATE_MODELS = [
    "patchtst",
    "itransformer",
    "timemixer",
    "gtr",
]

CANDIDATE_MODEL_CLASS_MAP = {
    "patchtst": "PatchTSTModel",
    "itransformer": "ITransformerModel",
    "timemixer": "TimeMixerModel",
    "gtr": "GTRModel",
}

CANDIDATE_MODULE_MAP = {
    "patchtst": "models.patchtst",
    "itransformer": "models.itransformer",
    "timemixer": "models.timemixer",
    "gtr": "models.gtr",
}

SELECTION_METRIC = "val_mae"
TIE_BREAKERS = ["val_rmse", "val_mse"]
EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    """Sanitise a string for use in directory names."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _dataset_path_parts(dataset_id: str) -> list[str]:
    """Keep nested dataset identifiers such as ett/ETTh1 separated."""
    parts = [part for part in str(dataset_id).replace(chr(92), "/").split("/") if part]
    return [_safe_name(part) for part in parts] or ["unknown_dataset"]


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Identity hash for selection unit
# ---------------------------------------------------------------------------

def compute_split_identity(
    *,
    dataset_id: str,
    target_column: str,
    input_len: int,
    pred_len: int,
    stride: int,
    seed: int,
    scaler_info: dict[str, Any],
    split_info: dict[str, Any],
) -> str:
    """Deterministic hash binding selection to data, target, config, and split."""
    payload = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "seed": int(seed),
        "scaler_mean": [float(x) for x in scaler_info.get("mean", [])],
        "scaler_scale": [float(x) for x in scaler_info.get("scale", [])],
        "split_train": split_info.get("windows", {}).get("train"),
        "split_val": split_info.get("windows", {}).get("val"),
        "split_test": split_info.get("windows", {}).get("test"),
        "split_rows_train": split_info.get("rows", {}).get("train"),
        "split_rows_val": split_info.get("rows", {}).get("val"),
        "split_rows_test": split_info.get("rows", {}).get("test"),
    }
    identity_json = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return _sha256_bytes(identity_json.encode("utf-8"))


# ---------------------------------------------------------------------------
# Pretrained directory layout
# ---------------------------------------------------------------------------

def teacher_dir(
    *,
    dataset_id: str,
    target_column: str,
    input_len: int,
    pred_len: int,
    stride: int,
    split_identity: str,
) -> Path:
    """Return the canonical per-config teacher directory."""
    return (
        PRETRAINED_ROOT.joinpath(*_dataset_path_parts(dataset_id))
        / _safe_name(target_column)
        / f"input_{input_len}_pred_{pred_len}_stride_{stride}"
        / f"split_{split_identity[:16]}"
        / "best_teacher"
    )


# ---------------------------------------------------------------------------
# Candidate result collection
# ---------------------------------------------------------------------------

def collect_candidate_results(
    *,
    model_checkpoint_dirs: dict[str, Path],
    model_run_status: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate validation metrics from the four candidates.

    Parameters
    ----------
    model_checkpoint_dirs : dict
        model_name -> Path to the model's artifact directory.
        Expected to contain checkpoints/best_model.pt and metrics/metrics.json.
    model_run_status : dict
        model_name -> run status dict with keys: status, metrics.

    Returns
    -------
    list[dict]
        One entry per candidate, sorted by ranking (best first).  Entries for
        failed models include the failure reason.
    """
    entries: list[dict[str, Any]] = []
    for model_name in CANDIDATE_MODELS:
        status_info = model_run_status.get(model_name, {})
        status = status_info.get("status", "UNKNOWN")
        metrics = status_info.get("metrics", {}) or {}
        ckpt_dir = model_checkpoint_dirs.get(model_name)
        ckpt_path = ckpt_dir / "checkpoints" / "best_model.pt" if ckpt_dir else None

        entry: dict[str, Any] = {
            "model_name": model_name,
            "status": status,
            "val_mae": metrics.get("MAE"),
            "val_rmse": metrics.get("RMSE"),
            "val_mse": metrics.get("MSE"),
            "checkpoint_path": str(ckpt_path) if ckpt_path and ckpt_path.exists() else None,
            "checkpoint_sha256": _sha256_file(ckpt_path) if ckpt_path and ckpt_path.exists() else None,
            "parameter_count": None,
            "training_runtime": status_info.get("train_time_sec"),
            "error_message": status_info.get("error_message", ""),
        }
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def select_best_teacher(
    candidates: list[dict[str, Any]],
    *,
    require_all: bool = True,
) -> dict[str, Any]:
    """Select the best teacher from candidate results.

    Rules:
    1. Lower val_mae is better.
    2. On near-tie (within EPSILON relative), compare val_rmse (lower better).
    3. Further tied → val_mse (lower better).
    4. Still tied → fixed candidate order (PatchTST → iTransformer → TimeMixer → GTR).

    Parameters
    ----------
    candidates : list[dict]
        Output of collect_candidate_results().
    require_all : bool
        If True (default), raise if any candidate failed.

    Returns
    -------
    dict
        Selection result with selected_model, ranking, and all candidate info.
    """
    failed = [c for c in candidates if c["status"] != "SUCCESS"]
    if failed and require_all:
        names = [c["model_name"] for c in failed]
        raise RuntimeError(
            f"Teacher selection requires all {len(CANDIDATE_MODELS)} candidates to succeed. "
            f"Failed: {names}.  Set require_all=False to select from remaining."
        )

    valid = [c for c in candidates if c["status"] == "SUCCESS"]
    if not valid:
        raise RuntimeError("No successful candidates available for teacher selection.")

    # Deterministic sort
    tie_order = {name: idx for idx, name in enumerate(CANDIDATE_MODELS)}

    def sort_key(entry: dict[str, Any]) -> tuple[float, float, float, int]:
        mae = float(entry.get("val_mae", float("inf")) or float("inf"))
        rmse = float(entry.get("val_rmse", float("inf")) or float("inf"))
        mse = float(entry.get("val_mse", float("inf")) or float("inf"))
        return (mae, rmse, mse, tie_order.get(entry["model_name"], 999))

    ranked = sorted(valid, key=sort_key)
    best = ranked[0]

    # Assign ranking and selected flag
    for idx, entry in enumerate(ranked):
        entry["ranking"] = idx + 1
        entry["selected"] = entry is best

    # Mark failed entries
    for entry in failed:
        entry["ranking"] = None
        entry["selected"] = False

    # Combine: ranked first, then failed
    all_ranked = ranked + failed

    selection = {
        "selected_model": best["model_name"],
        "selected_module": CANDIDATE_MODULE_MAP.get(best["model_name"]),
        "selected_model_class": CANDIDATE_MODEL_CLASS_MAP.get(best["model_name"]),
        "selection_metric": SELECTION_METRIC,
        "selected_validation_mae": best["val_mae"],
        "selected_validation_rmse": best["val_rmse"],
        "selected_validation_mse": best["val_mse"],
        "source_checkpoint_path": best["checkpoint_path"],
        "source_checkpoint_sha256": best["checkpoint_sha256"],
        "candidate_count": len(candidates),
        "candidate_success_count": len(valid),
        "candidate_failed_count": len(failed),
        "candidates": all_ranked,
    }
    return selection


# ---------------------------------------------------------------------------
# Persist teacher artifacts
# ---------------------------------------------------------------------------

def save_teacher_artifacts(
    *,
    tdir: Path,
    selection: dict[str, Any],
    dataset_id: str,
    dataset_path: str,
    target_column: str,
    input_len: int,
    pred_len: int,
    stride: int,
    seed: int,
    split_identity: str,
    scaler_info: dict[str, Any],
    code_version: str = "",
) -> dict[str, Any]:
    """Write all teacher artifacts atomically.  READY is created last.

    Returns a summary of written files.
    """
    tdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # ---- selection.json ---------------------------------------------------
    selection_payload = dict(selection)
    selection_payload.update({
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
        "target_column": target_column,
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "seed": int(seed),
        "split_identity": split_identity,
        "scaler_info": scaler_info,
        "created_at": _utc_stamp(),
        "code_commit": code_version or "unknown",
        "teacher_dir": str(tdir),
    })
    _json_dump(tdir / "selection.json", selection_payload)
    written.append("selection.json")

    # ---- candidate_ranking.csv --------------------------------------------
    import csv

    csv_path = tdir / "candidate_ranking.csv"
    tmp_csv = tdir / "candidate_ranking.csv.tmp"
    fieldnames = [
        "model_name", "status", "val_mae", "val_rmse", "val_mse",
        "checkpoint_path", "checkpoint_sha256", "parameter_count",
        "training_runtime", "ranking", "selected", "error_message",
    ]
    with tmp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in selection.get("candidates", []):
            writer.writerow(entry)
    os.replace(tmp_csv, csv_path)
    written.append("candidate_ranking.csv")

    # ---- validation_metrics.json ------------------------------------------
    val_metrics = {
        "selection_metric": SELECTION_METRIC,
        "selected_model": selection["selected_model"],
        "selected_val_mae": selection["selected_validation_mae"],
        "selected_val_rmse": selection["selected_validation_rmse"],
        "selected_val_mse": selection["selected_validation_mse"],
        "all_candidates": {
            c["model_name"]: {
                "val_mae": c.get("val_mae"),
                "val_rmse": c.get("val_rmse"),
                "val_mse": c.get("val_mse"),
                "status": c.get("status"),
                "ranking": c.get("ranking"),
                "selected": c.get("selected"),
            }
            for c in selection.get("candidates", [])
        },
    }
    _json_dump(tdir / "validation_metrics.json", val_metrics)
    written.append("validation_metrics.json")

    # ---- split_manifest.json ----------------------------------------------
    split_manifest = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "seed": int(seed),
        "split_identity": split_identity,
    }
    _json_dump(tdir / "split_manifest.json", split_manifest)
    written.append("split_manifest.json")

    # ---- scaler_reference.json --------------------------------------------
    _json_dump(tdir / "scaler_reference.json", scaler_info)
    written.append("scaler_reference.json")

    # ---- teacher_config.json ----------------------------------------------
    teacher_config = {
        "selected_model": selection["selected_model"],
        "selected_module": selection.get("selected_module"),
        "selected_model_class": selection.get("selected_model_class"),
        "source_checkpoint_path": selection.get("source_checkpoint_path"),
        "source_checkpoint_sha256": selection.get("source_checkpoint_sha256"),
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "dataset_id": dataset_id,
        "target_column": target_column,
    }
    _json_dump(tdir / "teacher_config.json", teacher_config)
    written.append("teacher_config.json")

    # ---- source_checkpoint.json -------------------------------------------
    source_info = {
        "original_checkpoint_path": selection.get("source_checkpoint_path"),
        "sha256": selection.get("source_checkpoint_sha256"),
        "selected_model": selection.get("selected_model"),
        "copied_at": _utc_stamp(),
    }
    _json_dump(tdir / "source_checkpoint.json", source_info)
    written.append("source_checkpoint.json")

    # ---- teacher_checkpoint.pt (copy from source) -------------------------
    src_ckpt = selection.get("source_checkpoint_path")
    if src_ckpt and Path(src_ckpt).exists():
        dst_ckpt = tdir / "teacher_checkpoint.pt"
        shutil.copy2(src_ckpt, dst_ckpt)
        written.append("teacher_checkpoint.pt")
    else:
        raise FileNotFoundError(
            f"Source checkpoint not found: {src_ckpt}. "
            f"Cannot create teacher checkpoint."
        )

    # ---- checksums.sha256 -------------------------------------------------
    checksums: dict[str, str] = {}
    for name in written:
        fpath = tdir / name
        if fpath.exists():
            checksums[name] = _sha256_file(fpath)
    _json_dump(tdir / "checksums.sha256", checksums)
    written.append("checksums.sha256")

    # ---- READY (atomic, last) ---------------------------------------------
    ready_path = tdir / "READY"
    ready_tmp = tdir / "READY.tmp"
    ready_payload = {
        "created_at": _utc_stamp(),
        "files": written,
        "checksums": checksums,
        "teacher_model": selection["selected_model"],
        "dataset_id": dataset_id,
        "target_column": target_column,
        "split_identity": split_identity,
    }
    ready_tmp.write_text(
        json.dumps(ready_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(ready_tmp, ready_path)
    written.append("READY")

    return {
        "teacher_dir": str(tdir),
        "files_written": written,
        "ready": True,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_teacher_selection(
    *,
    dataset_id: str,
    dataset_path: str,
    target_column: str,
    input_len: int,
    pred_len: int,
    stride: int,
    seed: int,
    scaler_info: dict[str, Any],
    split_info: dict[str, Any],
    model_checkpoint_dirs: dict[str, Path],
    model_run_status: dict[str, dict[str, Any]],
    require_all: bool = True,
    code_version: str = "",
) -> dict[str, Any]:
    """Run the full teacher selection pipeline.

    Returns a dict suitable for downstream consumption (e.g. 11_distill).
    """
    split_identity = compute_split_identity(
        dataset_id=dataset_id,
        target_column=target_column,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        seed=seed,
        scaler_info=scaler_info,
        split_info=split_info,
    )

    tdir = teacher_dir(
        dataset_id=dataset_id,
        target_column=target_column,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        split_identity=split_identity,
    )

    # Check if already done
    ready_marker = tdir / "READY"
    if ready_marker.exists():
        selection = json.loads((tdir / "selection.json").read_text(encoding="utf-8"))
        print(
            f"[TeacherSelection] READY found at {tdir} — reusing. "
            f"Selected: {selection.get('selected_model')}",
            flush=True,
        )
        return {
            "teacher_dir": str(tdir),
            "selection": selection,
            "split_identity": split_identity,
            "reused": True,
        }

    candidates = collect_candidate_results(
        model_checkpoint_dirs=model_checkpoint_dirs,
        model_run_status=model_run_status,
    )

    selection = select_best_teacher(candidates, require_all=require_all)

    save_result = save_teacher_artifacts(
        tdir=tdir,
        selection=selection,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        target_column=target_column,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        seed=seed,
        split_identity=split_identity,
        scaler_info=scaler_info,
        code_version=code_version,
    )

    print(
        f"[TeacherSelection] Selected teacher: {selection['selected_model']} "
        f"val_mae={selection['selected_validation_mae']:.6g} "
        f"dir={tdir}",
        flush=True,
    )
    return {
        "teacher_dir": str(tdir),
        "selection": selection,
        "split_identity": split_identity,
        "save_result": save_result,
        "reused": False,
    }


# ---------------------------------------------------------------------------
# Teacher loader (for 11_distill.py and other consumers)
# ---------------------------------------------------------------------------

def load_teacher_from_selection(
    teacher_dir_path: str | Path,
    device: torch.device | str | None = None,
) -> "nn.Module":
    """Instantiate and load the selected teacher model from a teacher directory.

    Reads selection.json to determine the model type, then reconstructs it
    using the standard model class and loads the checkpoint state_dict.
    """
    import torch

    tdir = Path(teacher_dir_path)
    selection_path = tdir / "selection.json"
    ckpt_path = tdir / "teacher_checkpoint.pt"

    if not selection_path.exists():
        raise FileNotFoundError(f"selection.json not found in {tdir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"teacher_checkpoint.pt not found in {tdir}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_name = selection["selected_model"]
    model_class_name = selection.get("selected_model_class")

    # Map model name to class
    if model_name == "patchtst":
        from models.patchtst import PatchTSTModel
        model_cls = PatchTSTModel
    elif model_name == "itransformer":
        from models.itransformer import ITransformerModel
        model_cls = ITransformerModel
    elif model_name == "timemixer":
        from models.timemixer import TimeMixerModel
        model_cls = TimeMixerModel
    elif model_name == "gtr":
        from models.gtr import GTRModel
        model_cls = GTRModel
    else:
        raise ValueError(f"Unknown teacher model: {model_name}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Load the full checkpoint to get config
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})

    # Reconstruct with saved config plus overrides
    config = dict(saved_config)
    config["device"] = str(device)
    # Ensure pred_len matches
    config["pred_len"] = selection.get("pred_len", config.get("pred_len", 96))

    model = model_cls(config)
    if model_name == "gtr":
        # GTR persists its train-only retrieval bank through its custom
        # checkpoint API rather than a conventional neural state_dict.
        model.load(ckpt_path, map_location=device)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Verify checkpoint integrity
    expected_sha = selection.get("source_checkpoint_sha256")
    if expected_sha:
        actual_sha = _sha256_file(ckpt_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Teacher checkpoint SHA256 mismatch: "
                f"expected={expected_sha[:16]}... actual={actual_sha[:16]}..."
            )

    print(
        f"[TeacherLoader] Loaded teacher model={model_name} "
        f"from={tdir} params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    return model


# ---------------------------------------------------------------------------
# Validation — ensure fairness
# ---------------------------------------------------------------------------

def validate_candidate_fairness(
    *,
    candidate_val_predictions: dict[str, np.ndarray],
    candidate_val_y_true: dict[str, np.ndarray],
    candidate_val_sample_ids: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Check that all candidates used identical validation data.

    Returns a report; raises RuntimeError on critical mismatch.
    """
    issues: list[str] = []
    model_names = list(candidate_val_predictions.keys())

    if len(model_names) < 2:
        return {"valid": True, "issues": [], "model_count": len(model_names)}

    ref = model_names[0]
    ref_samples = len(candidate_val_sample_ids.get(ref, []))
    ref_shape = candidate_val_predictions.get(ref, np.array([])).shape
    ref_y = candidate_val_y_true.get(ref)

    for name in model_names[1:]:
        n = len(candidate_val_sample_ids.get(name, []))
        shape = candidate_val_predictions.get(name, np.array([])).shape
        y = candidate_val_y_true.get(name)
        if n != ref_samples:
            issues.append(
                f"Sample count mismatch: {ref}={ref_samples} vs {name}={n}"
            )
        if shape != ref_shape:
            issues.append(
                f"Prediction shape mismatch: {ref}={ref_shape} vs {name}={shape}"
            )
        if ref_y is not None and y is not None:
            if not np.allclose(ref_y, y, rtol=1e-6, atol=1e-8):
                issues.append(f"y_true mismatch between {ref} and {name}")

    valid = len(issues) == 0
    report = {
        "valid": valid,
        "issues": issues,
        "model_count": len(model_names),
        "reference_model": ref,
        "reference_samples": ref_samples,
    }
    if not valid:
        raise RuntimeError(
            f"Teacher candidate fairness check failed: {'; '.join(issues)}"
        )
    return report


__all__ = [
    "CANDIDATE_MODELS",
    "CANDIDATE_MODEL_CLASS_MAP",
    "CANDIDATE_MODULE_MAP",
    "SELECTION_METRIC",
    "run_teacher_selection",
    "load_teacher_from_selection",
    "teacher_dir",
    "compute_split_identity",
    "validate_candidate_fairness",
    "PRETRAINED_ROOT",
]
