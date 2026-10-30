from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .timer import cfg, load_shared_timer, serializable_config, timer_base_prediction, timer_contract, unpack_xy
from utils.training_control import ValidationEarlyStopping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "distill"
CACHE_VERSION = 3


class _StudentResidualHead(nn.Module):
    """Local MLP-based residual head — replaces external TimerStudentResidualHead."""

    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(seq_len) + int(pred_len), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(pred_len)),
        )

    def forward(self, x: torch.Tensor, timer_pred: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x.squeeze(-1), timer_pred.squeeze(-1)], dim=1)
        return self.net(features).unsqueeze(-1)


class DistillNetwork(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = _StudentResidualHead(seq_len, pred_len, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, timer_pred: torch.Tensor) -> torch.Tensor:
        return self.head(x, timer_pred)


class FrozenPredictionCache:
    def __init__(self, cache_dir: Path, metadata: dict[str, Any]):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self.sample_ids = np.load(cache_dir / "sample_id.npy", mmap_mode="r")
        self.timer_pred = np.load(cache_dir / "timer_pred.npy", mmap_mode="r")
        self.teacher_pred = np.load(cache_dir / "teacher_pred.npy", mmap_mode="r")

    def _indices_for(self, sample_ids: torch.Tensor | np.ndarray) -> np.ndarray:
        ids = np.asarray(sample_ids.detach().cpu() if isinstance(sample_ids, torch.Tensor) else sample_ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            return ids
        positions = np.searchsorted(self.sample_ids, ids)
        if (positions >= len(self.sample_ids)).any() or not np.array_equal(np.asarray(self.sample_ids[positions], dtype=np.int64), ids):
            raise KeyError("batch sample_id is not present in Distill-LSTM frozen prediction cache")
        return positions.astype(np.int64, copy=False)

    def timer_for(self, sample_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        indices = self._indices_for(sample_ids)
        return torch.from_numpy(np.asarray(self.timer_pred[indices], dtype=np.float32))

    def teacher_for(self, sample_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        indices = self._indices_for(sample_ids)
        return torch.from_numpy(np.asarray(self.teacher_pred[indices], dtype=np.float32))


class DistillModel(nn.Module):
    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device(cfg(self.config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pred_len = int(cfg(self.config, "pred_len", 96))
        self.epochs = int(cfg(self.config, "epochs", 10))
        self.learning_rate = float(cfg(self.config, "learning_rate", 1e-4))
        self.grad_clip = float(cfg(self.config, "grad_clip", 1.0))
        self.log_every_batches = max(
            0,
            int(cfg(self.config, "log_every_batches", os.environ.get("DISTILL_LSTM_LOG_EVERY_BATCHES", 500))),
        )
        self.cache_dir = Path(cfg(self.config, "distill_cache_dir", DEFAULT_CACHE_DIR)).expanduser().resolve()
        self._file_identity_cache: dict[str, dict[str, Any] | None] = {}
        self._cache_records: dict[str, dict[str, Any]] = {}
        self.timer_model = load_shared_timer(self.device, self.pred_len, cfg(self.config, "timer_model_dir", None) or None)
        self.contract = timer_contract(self.pred_len, cfg(self.config, "timer_model_dir", None) or None)
        self.adapter = DistillNetwork(96, self.pred_len, int(cfg(self.config, "hidden_dim", 128)), float(cfg(self.config, "dropout", 0.1))).to(self.device)
        self.criterion = nn.MSELoss()
        self.teacher = self._load_teacher()

    def _load_teacher(self) -> nn.Module:
        teacher_dir = cfg(self.config, "teacher_dir", None)
        if teacher_dir and Path(str(teacher_dir)).exists():
            from exp.teacher_selection import load_teacher_from_selection

            teacher = load_teacher_from_selection(str(teacher_dir), self.device)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self._teacher_source = "dynamic_selection"
            self._teacher_dir = str(teacher_dir)
            return teacher

        raise FileNotFoundError(
            f"No teacher_dir configured for distill model. "
            f"Run teacher selection first (after patchtst/itransformer/timemixer/gtr)."
        )

    def _student(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_timer = timer_base_prediction(self.timer_model, x)
        residual = self.adapter(x, y_timer)
        return y_timer, residual, y_timer + residual

    def _file_identity(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        key = str(path)
        if key in self._file_identity_cache:
            return self._file_identity_cache[key]
        if not path.exists() or not path.is_file():
            result = {"path": str(path), "exists": False}
            self._file_identity_cache[key] = result
            return result
        resolved = path.resolve()
        stat = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result = {
            "path": str(path),
            "realpath": str(resolved),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
        }
        self._file_identity_cache[key] = result
        return result

    def _cache_identity(self, split_name: str, loader: Any) -> dict[str, Any]:
        split_info = cfg(self.config, "split_info", {}) or {}
        teacher_identity = {
            "source": getattr(self, "_teacher_source", "unknown"),
            "teacher_dir": getattr(self, "_teacher_dir", None),
        }
        if self._teacher_dir:
            selection_path = Path(self._teacher_dir) / "selection.json"
            if selection_path.exists():
                try:
                    sel = json.loads(selection_path.read_text(encoding="utf-8"))
                    teacher_identity["selected_model"] = sel.get("selected_model")
                    teacher_identity["source_checkpoint_sha256"] = sel.get("source_checkpoint_sha256")
                    teacher_identity["selection_metric"] = sel.get("selection_metric")
                    teacher_identity["split_identity"] = sel.get("split_identity")
                except Exception:
                    pass
        return {
            "cache_version": CACHE_VERSION,
            "model": "distill",
            "split": split_name,
            "raw_file": self._file_identity(cfg(self.config, "raw_path", None)),
            "processed_file": self._file_identity(cfg(self.config, "processed_path", None)),
            "target_column": cfg(self.config, "target_column", None),
            "input_len": int(cfg(self.config, "input_len", 96)),
            "pred_len": int(self.pred_len),
            "stride": int(cfg(self.config, "stride", 1)),
            "loader_dataset_len": int(len(loader.dataset)) if hasattr(loader, "dataset") else None,
            "split_info": split_info,
            "timer_contract": self.contract,
            "teacher_identity": teacher_identity,
        }

    def _cache_path_for(self, split_name: str, loader: Any) -> tuple[Path, dict[str, Any], str]:
        identity = self._cache_identity(split_name, loader)
        identity_text = json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))
        identity_hash = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{split_name}_{identity_hash}", identity, identity_hash

    def _cache_complete(self, path: Path, identity_hash: str) -> bool:
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            return False
        if not all((path / name).exists() for name in ("sample_id.npy", "timer_pred.npy", "teacher_pred.npy")):
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("identity_hash") != identity_hash or metadata.get("cache_version") != CACHE_VERSION:
                return False
            for name in ("timer_pred.npy", "teacher_pred.npy"):
                if not np.isfinite(np.load(path / name, mmap_mode="r")).all():
                    return False
        except Exception:
            return False
        return True

    @staticmethod
    def _sample_ids_from_batch(batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            value = batch.get("sample_id", batch.get("sample_ids", batch.get("index")))
            if value is None:
                raise ValueError("Distill-LSTM cache requires sample_id in each batch")
            return torch.as_tensor(value)
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            return torch.as_tensor(batch[2])
        raise ValueError("Distill-LSTM cache requires DataLoader batches shaped as (x, y, sample_id)")

    @staticmethod
    def _match_prediction_shape(pred: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if pred.ndim == 2 and reference.ndim == 3 and reference.shape[-1] == 1:
            pred = pred.unsqueeze(-1)
        elif pred.ndim == 3 and pred.shape[-1] == 1 and reference.ndim == 2:
            pred = pred.squeeze(-1)
        if tuple(pred.shape) == tuple(reference.shape):
            return pred
        if pred.shape[0] == reference.shape[0] and pred[0].numel() == reference[0].numel():
            return pred.reshape(reference.shape)
        raise ValueError(f"{name} shape {tuple(pred.shape)} cannot be aligned to target shape {tuple(reference.shape)}")

    def _build_cache(self, path: Path, identity: dict[str, Any], identity_hash: str, split_name: str, loader: Any) -> dict[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        batch_size = int(getattr(loader, "batch_size", cfg(self.config, "batch_size", 128)) or 128)
        if not hasattr(loader, "dataset"):
            raise ValueError("Distill-LSTM cache requires a DataLoader with a dataset")
        stable_loader = DataLoader(loader.dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        total_rows = int(len(loader.dataset))
        sample_mm: np.ndarray | None = None
        timer_mm: np.ndarray | None = None
        teacher_mm: np.ndarray | None = None
        offset = 0
        started = time.perf_counter()
        print(f"[Distill-LSTM] building frozen cache split={split_name} rows={total_rows} path={path}", flush=True)
        self.timer_model.eval()
        self.teacher.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(stable_loader, start=1):
                x, y = unpack_xy(batch)
                sample_ids = self._sample_ids_from_batch(batch).detach().cpu().numpy().astype(np.int64)
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                teacher_pred = self._match_prediction_shape(self.teacher(x), y, "teacher_pred").detach().cpu().numpy().astype(np.float32)
                timer_pred = self._match_prediction_shape(timer_base_prediction(self.timer_model, x), y, "timer_pred").detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(teacher_pred).all():
                    raise FloatingPointError(f"teacher prediction contains NaN or Inf in {split_name} cache batch {batch_index}")
                if not np.isfinite(timer_pred).all():
                    raise FloatingPointError(f"Timer prediction contains NaN or Inf in {split_name} cache batch {batch_index}")
                rows = int(sample_ids.shape[0])
                if sample_mm is None:
                    sample_mm = np.lib.format.open_memmap(path / "sample_id.npy", mode="w+", dtype=np.int64, shape=(total_rows,))
                    timer_mm = np.lib.format.open_memmap(path / "timer_pred.npy", mode="w+", dtype=np.float32, shape=(total_rows, *timer_pred.shape[1:]))
                    teacher_mm = np.lib.format.open_memmap(path / "teacher_pred.npy", mode="w+", dtype=np.float32, shape=(total_rows, *teacher_pred.shape[1:]))
                sample_mm[offset : offset + rows] = sample_ids
                timer_mm[offset : offset + rows] = timer_pred
                teacher_mm[offset : offset + rows] = teacher_pred
                offset += rows
                if self.log_every_batches > 0 and (batch_index == 1 or batch_index % self.log_every_batches == 0):
                    elapsed = time.perf_counter() - started
                    print(f"[Distill-LSTM] cache split={split_name} batch={batch_index}/{len(stable_loader)} rows={offset}/{total_rows} elapsed={elapsed:.1f}s", flush=True)
        if offset != total_rows:
            raise ValueError(f"cache row count mismatch for {split_name}: wrote={offset} expected={total_rows}")
        sample_values = np.asarray(sample_mm[:], dtype=np.int64)
        order = np.argsort(sample_values, kind="stable")
        if not np.array_equal(order, np.arange(total_rows)):
            sample_mm[:] = sample_mm[order]
            timer_mm[:] = timer_mm[order]
            teacher_mm[:] = teacher_mm[order]
        del sample_mm, timer_mm, teacher_mm
        metadata = {
            "cache_version": CACHE_VERSION,
            "identity_hash": identity_hash,
            "identity": identity,
            "split": split_name,
            "rows": total_rows,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build_seconds": time.perf_counter() - started,
            "files": {
                "sample_id": "sample_id.npy",
                "timer_pred": "timer_pred.npy",
                "teacher_pred": "teacher_pred.npy",
            },
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return metadata

    def _frozen_cache(self, loader: Any, split_name: str) -> FrozenPredictionCache:
        path, identity, identity_hash = self._cache_path_for(split_name, loader)
        hit = self._cache_complete(path, identity_hash)
        if hit:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            print(f"[Distill-LSTM] frozen cache hit split={split_name} path={path}", flush=True)
        else:
            metadata = self._build_cache(path, identity, identity_hash, split_name, loader)
        record = {
            "split": split_name,
            "cache_hit": bool(hit),
            "path": str(path),
            "identity_hash": identity_hash,
            "rows": metadata.get("rows"),
            "build_seconds": metadata.get("build_seconds", 0.0 if hit else None),
        }
        self._cache_records[split_name] = record
        return FrozenPredictionCache(path, metadata)

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        train_cache = self._frozen_cache(train_loader, "train")
        val_cache = self._frozen_cache(val_loader, "val") if val_loader is not None else None
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.learning_rate)
        history: list[dict[str, float]] = []
        control = ValidationEarlyStopping(self, self.config)
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        train_started = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            epoch_started = time.perf_counter()
            self.adapter.train()
            total = hard_total = soft_total = 0.0
            count = 0
            print(f"[Distill-LSTM] epoch={epoch}/{self.epochs} start batches={total_batches or 'unknown'}", flush=True)
            for batch_index, batch in enumerate(train_loader, start=1):
                x, y = unpack_xy(batch)
                sample_ids = self._sample_ids_from_batch(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                y_timer = train_cache.timer_for(sample_ids).to(self.device).float()
                teacher_pred = train_cache.teacher_for(sample_ids).to(self.device).float()
                residual = self.adapter(x, y_timer)
                student = y_timer + residual
                hard = self.criterion(student, y)
                soft = self.criterion(student, teacher_pred)
                loss = 0.5 * hard + 0.5 * soft
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Distill-LSTM loss contains NaN or Inf at epoch {epoch}, batch {batch_index}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
                optimizer.step()
                batch_size = int(y.shape[0])
                total += float(loss.detach().cpu()) * batch_size
                hard_total += float(hard.detach().cpu()) * batch_size
                soft_total += float(soft.detach().cpu()) * batch_size
                count += batch_size
                should_log_batch = self.log_every_batches > 0 and (
                    batch_index == 1
                    or batch_index % self.log_every_batches == 0
                    or (total_batches is not None and batch_index == total_batches)
                )
                if should_log_batch:
                    elapsed = time.perf_counter() - epoch_started
                    avg_batch = elapsed / max(batch_index, 1)
                    remaining_batches = max((total_batches or batch_index) - batch_index, 0)
                    eta = avg_batch * remaining_batches
                    mem = ""
                    if torch.cuda.is_available() and self.device.type == "cuda":
                        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
                        mem = f" cuda_alloc_gb={allocated:.2f} cuda_reserved_gb={reserved:.2f}"
                    print(
                        f"[Distill-LSTM] epoch={epoch}/{self.epochs} batch={batch_index}/{total_batches or '?'} "
                        f"loss={float(loss.detach().cpu()):.6g} elapsed={elapsed:.1f}s avg_batch={avg_batch:.3f}s eta={eta:.1f}s{mem}",
                        flush=True,
                    )
            row = {"epoch": float(epoch), "train_loss": total / max(count, 1), "train_hard_loss": hard_total / max(count, 1), "train_soft_loss": soft_total / max(count, 1)}
            if val_loader is not None and val_cache is not None:
                row.update(self._evaluate_validation(val_loader, val_cache))
                is_best = control.update(row["val_mae"], epoch)
                control.annotate(row, is_best=is_best)
            history.append(row)
            elapsed_total = time.perf_counter() - train_started
            epoch_seconds = time.perf_counter() - epoch_started
            eta_total = (elapsed_total / epoch) * (self.epochs - epoch) if epoch > 0 else 0.0
            print(
                f"[Distill-LSTM] epoch={epoch}/{self.epochs} done train_loss={row['train_loss']:.6g} "
                f"val_loss={row.get('val_loss', float('nan')):.6g} val_mae={row.get('val_mae', float('nan')):.6g} "
                f"best_epoch={control.best_epoch} patience={control.bad_epochs} "
                f"epoch_seconds={epoch_seconds:.1f} elapsed={elapsed_total:.1f}s eta={eta_total:.1f}s",
                flush=True,
            )
            if val_loader is not None and control.stopped_early:
                print(f"[Distill-LSTM] early stopping at epoch={epoch}; best_epoch={control.best_epoch}", flush=True)
                break
        if val_loader is not None:
            control.restore_best()
        return {
            "history": history,
            "training_control": control.summary(len(history)),
            "uses_timer0_base": True,
            "teacher_source": getattr(self, "_teacher_source", "unknown"),
            "teacher_dir": getattr(self, "_teacher_dir", None),
            "teacher": str(self._teacher_dir),
            "timer_contract": self.contract,
            "timer_frozen": True,
            "log_every_batches": self.log_every_batches,
            "frozen_prediction_cache": self._cache_records,
        }

    @torch.inference_mode()
    def _evaluate_validation(self, loader: Any, cache: FrozenPredictionCache) -> dict[str, float]:
        self.adapter.eval()
        squared_error = 0.0
        absolute_error = 0.0
        count = 0
        for batch in loader:
            x, y = unpack_xy(batch)
            sample_ids = self._sample_ids_from_batch(batch)
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            y_timer = cache.timer_for(sample_ids).to(self.device).float()
            prediction = y_timer + self.adapter(x, y_timer)
            error = prediction - y
            if not torch.isfinite(error).all():
                raise FloatingPointError("Distill-LSTM validation prediction contains NaN or Inf")
            squared_error += float(error.square().sum().cpu())
            absolute_error += float(error.abs().sum().cpu())
            count += int(error.numel())
        if count == 0:
            raise ValueError("Distill-LSTM validation loader produced no target elements")
        return {"val_loss": squared_error / count, "val_mae": absolute_error / count}

    def forward(self, x: torch.Tensor, y_reference: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()
        return self._student(x)[2]

    def predict(self, test_loader: Any, *, return_numpy: bool = True) -> np.ndarray | torch.Tensor:
        test_cache = self._frozen_cache(test_loader, "test")
        self.adapter.eval()
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in test_loader:
                x = torch.as_tensor(batch[0], device=self.device).float()
                sample_ids = self._sample_ids_from_batch(batch)
                y_timer = test_cache.timer_for(sample_ids).to(self.device).float()
                residual = self.adapter(x, y_timer)
                preds.append((y_timer + residual).detach().cpu())
        out = torch.cat(preds, dim=0)
        return out.numpy() if return_numpy else out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "adapter_state_dict": self.adapter.state_dict(),
            "config": serializable_config(self.config),
            "timer_contract": self.contract,
            "uses_timer0_base": True,
            "teacher_source": getattr(self, "_teacher_source", "unknown"),
            "teacher_dir": getattr(self, "_teacher_dir", None),
            "frozen_prediction_cache": self._cache_records,
        }, path)

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> "DistillModel":
        payload = torch.load(path, map_location=map_location or self.device)
        self.adapter.load_state_dict(payload["adapter_state_dict"])
        return self


__all__ = ["DistillNetwork", "DistillModel"]
