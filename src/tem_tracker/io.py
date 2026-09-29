from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("tracker", "registration", "calibration", "output", "preprocessing"):
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    for key in ("acquisition_fps", "nm_per_pixel"):
        val = config["calibration"].get(key)
        if val is not None and (not np.isfinite(val) or val <= 0):
            raise ValueError(f"{key} must be null or a positive number")
    if config["tracker"]["search_range"] <= 0 or config["tracker"]["memory"] < 0:
        raise ValueError("search_range must be positive and memory nonnegative")
    return config


def load_video(path: str | Path, max_memory_mb: int = 1024) -> tuple[list[np.ndarray], dict]:
    path = Path(path).resolve(strict=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, timestamps, total = [], [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            total += gray.nbytes
            if total > max_memory_mb * 1024**2:
                raise MemoryError("Video exceeds configured memory cap; increase output.max_memory_mb for offline processing")
            frames.append(gray)
            timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
    finally:
        cap.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video has no decodable frames or no valid playback fps")
    if declared > 0 and declared != len(frames):
        raise ValueError(f"Video decoded {len(frames)} of declared {declared} frames; refusing incomplete analysis")
    if len({f.shape for f in frames}) != 1:
        raise ValueError("Variable frame dimensions are not supported")
    h, w = frames[0].shape
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        timestamps = (np.arange(len(frames)) / fps).tolist()
        source = "frame_index_div_container_fps"
    else:
        source = "decoder_timestamps"
    return frames, {"input": str(path), "frames": len(frames), "width": w, "height": h,
                    "container_fps": fps, "duration_video_s": len(frames)/fps,
                    "video_timestamps_s": timestamps, "timestamp_source": source,
                    "sha256": sha256(path)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_seeds(data: dict, shape: tuple[int, int], nframes: int) -> list[dict]:
    seeds = data.get("particles", [])
    if not seeds:
        raise ValueError("Select at least one particle")
    h, w = shape
    seen = set()
    result = []
    for item in seeds:
        s = {k: float(item[k]) for k in ("x", "y", "radius")}
        if not all(np.isfinite(list(s.values()))):
            raise ValueError("Particle coordinates must be finite")
        if int(item["id"]) != float(item["id"]) or int(item.get("frame",0)) != float(item.get("frame",0)):
            raise ValueError("Particle IDs and frames must be integers")
        s.update(id=int(item["id"]), frame=int(item.get("frame", 0)))
        if s["id"] < 1 or not 0 <= s["frame"] < nframes or not (0 <= s["x"] < w and 0 <= s["y"] < h):
            raise ValueError(f"Invalid ID/frame/position: {s}")
        if not 4 <= s["radius"] <= min(h, w)/3:
            raise ValueError("ROI radius must be at least 4 px and at most one third of frame size")
        if (s["id"], s["frame"]) in seen:
            raise ValueError("Duplicate ID at the same anchor frame")
        seen.add((s["id"], s["frame"]))
        result.append(s)
    return sorted(result, key=lambda s: (s["frame"], s["id"]))


def environment() -> dict:
    names = ("numpy", "scipy", "pandas", "opencv-python-headless", "opencv-python", "trackpy",
             "matplotlib", "PyYAML", "torch", "ultralytics", "imageio-ffmpeg")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "opencv": cv2.__version__, "device": "cpu", "cuda_used": False}


def save_json(path: Path, value: Any) -> None:
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
