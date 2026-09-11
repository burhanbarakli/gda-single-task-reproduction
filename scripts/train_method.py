#!/usr/bin/env python3
"""Run one frozen G5 method with path rewriting, preflight and resume support."""
import argparse
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from verify_dataset import verify as verify_dataset

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["ot-sim2real", "MMD", "cotrain", "source_only", "target_only"]
DEFAULT_RUN_IDS = {
    "ot-sim2real": "g5_ot-sim2real_seed1_reproduction",
    "MMD": "g5_MMD_seed1_reproduction",
    "cotrain": "g5_cotrain_seed1_reproduction",
    "source_only": "g5_source_only_seed1_reproduction",
    "target_only": "g5_target_only_seed1_reproduction",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rewrite_paths(value, old_root: str, new_root: str):
    if isinstance(value, dict):
        return {key: rewrite_paths(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, old_root, new_root) for item in value]
    if isinstance(value, str) and value.startswith(old_root + "/"):
        return new_root.rstrip("/") + value[len(old_root):]
    return value


def build_config(method: str, data_root: Path, output_root: Path, run_id: str) -> dict:
    path = ROOT / "config/reference" / f"{method}_seed1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config = rewrite_paths(config, "/home/aistack/gda-data", str(data_root))
    config["experiment"]["name"] = run_id
    config["train"]["output_dir"] = str(output_root)
    config["train"]["cuda"] = True
    return config


def assert_protocol(config: dict) -> None:
    expected = json.loads((ROOT / "manifests/protocol.json").read_text())["training"]
    checks = {
        "seed": config["train"]["seed"] == expected["seed"],
        "epochs": config["train"]["num_epochs"] == expected["epochs"],
        "steps": config["experiment"]["epoch_every_n_steps"] == expected["steps_per_epoch"],
        "batch": config["train"]["batch_size"] == expected["batch_size"],
        "workers": config["train"]["num_data_workers"] == expected["workers"],
        "sequence": config["train"]["seq_length"] == expected["sequence_length"],
        "frame_stack": config["train"]["frame_stack"] == expected["frame_stack"],
        "checkpoint": config["experiment"]["save"]["every_n_epochs"] == expected["checkpoint_every_epochs"],
        "rollout_off": not config["experiment"]["rollout"]["enabled"],
    }
    if not all(checks.values()):
        raise RuntimeError("Donmuş protokol değişmiş: " + json.dumps(checks))


def check_repositories() -> dict:
    protocol = json.loads((ROOT / "manifests/protocol.json").read_text())
    expected = {"ot-sim2real": protocol["official_commit"], **protocol["dependency_commits"]}
    actual = {}
    for name, commit in expected.items():
        value = subprocess.check_output(["git", "-C", str(ROOT / "vendor" / name), "rev-parse", "HEAD"], text=True).strip()
        if value != commit:
            raise RuntimeError(f"{name} commit farklı: {value} != {commit}")
        actual[name] = value
    patch = (ROOT / "patches/0001-demo-rights-and-seeds.patch").read_bytes()
    diff = subprocess.check_output(["git", "-C", str(ROOT / "vendor/ot-sim2real"), "diff", "--no-ext-diff", "--binary"])
    if diff != patch:
        raise RuntimeError("Resmî kod çalışma ağacı kayıtlı patch ile uyuşmuyor")
    return actual


def gpu_snapshot(index: int) -> dict:
    raw = subprocess.check_output(["nvidia-smi", f"--id={index}", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [item.strip() for item in raw.split(",")]
    return {"name": fields[0], "total_mib": int(fields[1]), "free_mib": int(fields[2]), "used_mib": int(fields[3]), "utilization_percent": int(fields[4])}


@contextmanager
def run_lock(output_root: Path, run_id: str):
    lock_path = output_root / ".locks" / f"{run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{run_id} zaten çalışıyor") from exc
        stream.write(str(os.getpid()))
        yield


def locate_run(output_root: Path, run_id: str) -> Path | None:
    base = output_root / run_id
    if not base.exists():
        return None
    candidates = sorted(path for path in base.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Tek zaman damgalı koşu dizini bekleniyordu: {candidates}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-allocator-cap-mib", type=int, default=9216)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Linux veya WSL2 içinde çalıştırın.")
    data_root, output_root = args.data_root.expanduser().resolve(), args.output_root.expanduser().resolve()
    run_id = args.run_id or DEFAULT_RUN_IDS[args.method]
    config = build_config(args.method, data_root, output_root, run_id)
    assert_protocol(config)
    data_report = verify_dataset(data_root)
    repositories = check_repositories()
    gpu = gpu_snapshot(args.gpu_index)
    required_free = args.gpu_allocator_cap_mib + 1024
    if gpu["free_mib"] < required_free:
        raise SystemExit(f"GPU boş belleği yetersiz: {gpu['free_mib']} MiB < {required_free} MiB")
    preflight = {"ok": True, "method": args.method, "run_id": run_id, "data": data_report, "repositories": repositories, "gpu": gpu}
    if args.validate_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from robomimic.config import config_factory
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch tarafından görünmüyor")
    full = config_factory(config["algo_name"])
    with full.values_unlocked():
        full.update(config)
    full.lock()
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    result_path = artifacts / f"{run_id}-result.json"
    executed_path = artifacts / f"{run_id}-config.json"
    executed_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {**preflight, "status": "running", "started_at": now(), "config": str(executed_path), "config_sha256": sha256(executed_path)}
    started = time.perf_counter()
    exit_code = 1
    with run_lock(output_root, run_id):
        run_dir = locate_run(output_root, run_id)
        resume = bool(run_dir and ((run_dir / "last.pth").is_file() or (run_dir / "last_bak.pth").is_file()))
        try:
            if multiprocessing.get_start_method(allow_none=True) is None:
                multiprocessing.set_start_method("fork")
            if multiprocessing.get_start_method() != "fork":
                raise RuntimeError("multiprocessing yöntemi fork olmalı")
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(args.gpu_allocator_cap_mib * 1024**2 / total)
            module_name = "robomimic.scripts.ot_sim2real.ot_train" if "ot" in config["train"] else "robomimic.scripts.train"
            report.update(resume=resume, entry=module_name + ".train")
            if args.method == "ot-sim2real":
                importlib.import_module(module_name).train(full, torch.device("cuda:0"), resume=resume)
            else:
                from observation_io_candidate import observation_prefix_reads
                with observation_prefix_reads():
                    importlib.import_module(module_name).train(full, torch.device("cuda:0"), resume=resume)
            run_dir = locate_run(output_root, run_id)
            checkpoint = run_dir / "models/model_epoch_500.pth"
            if not checkpoint.is_file():
                raise RuntimeError(f"Nihai checkpoint bulunamadı: {checkpoint}")
            report.update(status="completed", completed_at=now(), run_dir=str(run_dir), final_checkpoint=str(checkpoint), final_checkpoint_bytes=checkpoint.stat().st_size, final_checkpoint_sha256=sha256(checkpoint), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            exit_code = 0
        except KeyboardInterrupt:
            report.update(status="interrupted", interrupted_at=now())
            exit_code = 130
        except BaseException:
            report.update(status="failed", failed_at=now(), error=traceback.format_exc())
        finally:
            report.update(seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(report, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

