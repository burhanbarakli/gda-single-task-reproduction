#!/usr/bin/env python3
"""Run the frozen 100-rollout final test for one epoch-500 checkpoint."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ot-sim2real", "MMD", "cotrain", "source_only", "target_only"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/final"))
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    out = args.results_root.expanduser().resolve() / args.method
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "evaluation.json"
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("status") == "completed":
            print(json.dumps(prior, indent=2)); return 0
        raise SystemExit(f"Kısmi test korunuyor; inceleyip yeni sonuç dizini seçin: {report_path}")
    lock = args.results_root.expanduser().resolve() / "evaluation.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit("Başka bir final değerlendirme çalışıyor") from exc
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    report = {"method": args.method, "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "episodes": []}

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    try:
        import h5py
        import imageio.v2 as imageio
        import numpy as np
        import torch
        import robomimic.envs.robosuite
        from robomimic.utils import file_utils as FileUtils, env_utils as EnvUtils
        from robomimic.scripts.run_trained_agent import rollout

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA görünmüyor")
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(2048 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if raw["variable_state"]["epoch"] != 500:
            raise RuntimeError("Yalnız epoch 500 checkpoint değerlendirilebilir")
        policy, raw = FileUtils.policy_from_checkpoint(ckpt_dict=raw, device=torch.device("cuda:0"))
        cfg, _ = FileUtils.config_from_checkpoint(ckpt_dict=raw)
        report.update(status="evaluating", policy_class=type(policy.policy).__name__)
        save()
        for domain, seed_start in [("down", 900000), ("up", 910000)]:
            metadata = data_root / f"sim_demos/stack_r2l_{domain}/table-wood_120x160.hdf5"
            with h5py.File(metadata, "r") as handle:
                env_meta = json.loads(handle["data"].attrs["env_args"])
            env = EnvUtils.create_env_from_metadata(env_meta, use_image_obs=True, render_offscreen=True)
            env = EnvUtils.wrap_env_from_config(env, config=cfg)
            try:
                for seed in range(seed_start, seed_start + 50):
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    started = time.perf_counter()
                    video = out / f"{domain}-{seed}.mp4"
                    writer_context = imageio.get_writer(str(video), fps=10, macro_block_size=1) if seed == seed_start else nullcontext(None)
                    with writer_context as writer:
                        stats, trajectory = rollout(policy, env, horizon=500, video_writer=writer, video_skip=2, camera_names=["agentview"], camera_height=120, camera_width=160)
                    if not np.isfinite(trajectory["actions"]).all() or not np.isfinite(trajectory["states"]).all():
                        raise RuntimeError("Sonlu olmayan rollout değeri")
                    initial = np.asarray(trajectory["initial_state_dict"]["states"])
                    paired = args.results_root.expanduser().resolve() / "initial_states" / f"{domain}-{seed}.npy"
                    paired.parent.mkdir(exist_ok=True)
                    if paired.exists() and not np.array_equal(np.load(paired, allow_pickle=False), initial):
                        raise RuntimeError("Yöntemler arasında başlangıç durumu farklı")
                    if not paired.exists(): np.save(paired, initial, allow_pickle=False)
                    episode = {"domain": domain, "reset_seed": seed, "statistics": stats, "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(), "seconds": time.perf_counter() - started}
                    report["episodes"].append(episode); save(); print(json.dumps(episode), flush=True)
            finally:
                env.unwrapped.env.close()
        report["domains"] = {}
        for domain in ("down", "up"):
            rows = [row for row in report["episodes"] if row["domain"] == domain]
            successes = sum(int(row["statistics"]["Success_Rate"]) for row in rows)
            report["domains"][domain] = {"successes": successes, "rollouts": 50, "success_rate": successes / 50}
        report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(); return 0
    except BaseException:
        report.update(status="failed", error=traceback.format_exc()); save(); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
