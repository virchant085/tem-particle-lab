"""Export observed particle paths and sampled ROI circles in one time-coded image.

This renderer only reads recorded raw-image coordinates. It never associates,
interpolates, smooths, or converts video frame indices to experimental time.
"""
from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.patheffects as pe


_REQUIRED = {"frame", "track_id", "center_x", "center_y", "radius_px", "observed", "status"}
_TIME_COLORS = ["#4979ff", "#27bce5", "#79d8ad", "#ffe05a"]


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name}必须是大于等于 {minimum} 的整数")
    return int(value)


def _observed(value) -> bool:
    # Avoid bool('False') and bool(np.nan), which both silently become True.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if pd.isna(value):
        return False
    raise ValueError("observed 列必须使用布尔值或 0/1")


def _prepare_tracks(tracks: pd.DataFrame, track_ids: list[int], frame_start: int,
                    frame_end: int, circle_interval: int) -> list[dict]:
    """Build drawing geometry, retaining gaps and correction boundaries."""
    if not isinstance(tracks, pd.DataFrame) or not _REQUIRED.issubset(tracks.columns):
        raise ValueError("轨迹数据缺少 frame、track_id、center_x、center_y、radius_px、observed 或 status 列")
    if not isinstance(track_ids, list) or not track_ids:
        raise ValueError("请至少选择一个颗粒 ID")
    ids = [_integer(ident, "颗粒 ID") for ident in track_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("颗粒 ID 不能重复")
    table = tracks.copy(deep=True)
    for name in ("frame", "track_id"):
        values = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
            raise ValueError(f"轨迹 {name} 列必须是非负整数")
        table[name] = values.astype(np.int64)
    unknown = sorted(set(ids) - set(table.track_id))
    if unknown:
        raise ValueError("找不到颗粒 ID：" + "、".join(map(str, unknown)))
    table = table[table.track_id.isin(ids) & table.frame.between(frame_start, frame_end)].copy()
    if table.duplicated(["track_id", "frame"]).any():
        raise ValueError("同一颗粒在同一帧存在重复记录，无法确定位置")
    for name in ("center_x", "center_y", "radius_px"):
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table["observed"] = table.observed.map(_observed).astype(bool)
    valid = np.isfinite(table[["center_x", "center_y", "radius_px"]].to_numpy(dtype=float)).all(axis=1)
    table["_valid"] = table.observed & valid & table.radius_px.gt(0)
    particles = []
    for ident in ids:
        group = table[table.track_id == ident].sort_values("frame")
        points = group[group._valid]
        segments, segment_frames, segment_times = [], [], []
        previous = None
        for _, row in group.iterrows():
            if not row["_valid"]:
                previous = None
                continue
            if (previous is not None and row.frame == previous.frame + 1
                    and row.status != "manual_anchor"):
                segments.append([[float(previous.center_x), float(previous.center_y)],
                                 [float(row.center_x), float(row.center_y)]])
                segment_frames.append([int(previous.frame), int(row.frame)])
                segment_times.append((int(previous.frame) + int(row.frame)) / 2)
            previous = row
        if points.empty:
            samples = points
        else:
            # All IDs share the start-anchored grid. Missing grid slots stay empty.
            selected = (points.frame - frame_start).mod(circle_interval).eq(0)
            selected |= points.frame.isin([points.frame.iloc[0], points.frame.iloc[-1]])
            samples = points[selected]
        particles.append({"track_id": ident, "points": points, "samples": samples,
                          "segments": segments, "segment_frames": segment_frames,
                          "segment_times": segment_times})
    if not any(len(particle["points"]) for particle in particles):
        raise ValueError("所选颗粒在该帧范围内没有有效观测，请更换颗粒或帧范围")
    return particles


def _time_scale(frame_start: int, frame_end: int):
    if frame_start == frame_end:
        return ListedColormap([_TIME_COLORS[0]]), Normalize(frame_start - .5, frame_end + .5)
    return LinearSegmentedColormap.from_list("particle_time", _TIME_COLORS), Normalize(frame_start, frame_end)


def render_time_traces(background: np.ndarray, tracks: pd.DataFrame, output: Path, *,
                       track_ids: list[int], frame_start: int, frame_end: int,
                       circle_interval: int = 5) -> dict:
    """Save a PNG of paths plus time-colored ROI circle trails; return provenance.

    ``background`` is the image at ``frame_start`` in grayscale or RGB(A).
    Frames are zero based and the requested range is inclusive. A positive,
    finite radius is required for a valid observation. Invalid rows break paths.
    All parameters are checked before writing; ``tracks`` is never modified.
    """
    frame_start = _integer(frame_start, "起始帧")
    frame_end = _integer(frame_end, "结束帧")
    circle_interval = _integer(circle_interval, "残影间隔", 1)
    if frame_end < frame_start:
        raise ValueError("结束帧不能早于起始帧")
    if (not isinstance(background, np.ndarray) or background.size == 0
            or background.ndim not in (2, 3)
            or (background.ndim == 3 and background.shape[2] not in (3, 4))
            or not np.issubdtype(background.dtype, np.number)
            or not np.isfinite(background).all()):
        raise ValueError("背景必须是有效的灰度或 RGB 图像")
    particles = _prepare_tracks(tracks, track_ids, frame_start, frame_end, circle_interval)
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("时间轨迹图输出路径必须以 .png 结尾")
    cmap, norm = _time_scale(frame_start, frame_end)
    metadata = {
        "frame_start": frame_start, "frame_end": frame_end, "circle_interval": circle_interval,
        "track_ids": [int(ident) for ident in track_ids], "time_basis": "frame_index",
        "coordinate_basis": "raw_image_pixels", "background_frame": frame_start,
        "circle_basis": "recorded_roi_radius_not_segmentation", "interpolated_count": 0,
        "observed_count": 0, "circle_count": 0, "segment_count": 0,
        "missing_ids": [], "particles": [],
    }
    # The object-oriented Agg canvas does not register figures or mutate pyplot state.
    fig = Figure(figsize=(10, 10), dpi=180, facecolor="#f7f8fa")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([.09, .235, .70, .665])
    ax.imshow(background, cmap="gray" if background.ndim == 2 else None,
              interpolation="nearest", origin="upper")
    height, width = background.shape[:2]
    ax.set(xlim=(-.5, width - .5), ylim=(height - .5, -.5),
           xlabel="x / pixels", ylabel="y / pixels")
    ax.tick_params(labelsize=9, colors="#526070")
    for spine in ax.spines.values():
        spine.set_color("#8895a3")
    halo = [pe.Stroke(linewidth=4.2, foreground="#101820", alpha=.8), pe.Normal()]
    text_halo = [pe.Stroke(linewidth=3, foreground="#102029"), pe.Normal()]
    for particle in particles:
        ident, points, samples = particle["track_id"], particle["points"], particle["samples"]
        record = {"track_id": ident, "observed_count": len(points), "circle_count": len(samples),
                  "first_frame": None, "last_frame": None,
                  "sampled_frames": [int(frame) for frame in samples.frame],
                  "segments": particle["segment_frames"]}
        metadata["particles"].append(record)
        metadata["observed_count"] += len(points)
        metadata["circle_count"] += len(samples)
        metadata["segment_count"] += len(particle["segments"])
        if points.empty:
            metadata["missing_ids"].append(ident)
            continue
        first, last = points.iloc[0], points.iloc[-1]
        record.update(first_frame=int(first.frame), last_frame=int(last.frame))
        for row in samples.itertuples():
            circle = Circle((row.center_x, row.center_y), row.radius_px, fill=False,
                            edgecolor=cmap(norm(row.frame)), linewidth=1.65, alpha=.72, zorder=3)
            circle.set_path_effects([pe.Stroke(linewidth=2.9, foreground="#101820", alpha=.5), pe.Normal()])
            ax.add_patch(circle)
        if particle["segments"]:
            line = LineCollection(particle["segments"], cmap=cmap, norm=norm, linewidth=2.15, zorder=4)
            line.set_array(np.asarray(particle["segment_times"]))
            line.set_path_effects(halo)
            ax.add_collection(line)
        ax.scatter(points.center_x, points.center_y, c=points.frame, cmap=cmap, norm=norm,
                   s=15, edgecolors="#16232a", linewidths=.45, zorder=5)
        ax.scatter([first.center_x], [first.center_y], marker="o", s=67,
                   facecolors="none", edgecolors="white", linewidths=1.7, zorder=7)
        ax.scatter([last.center_x], [last.center_y], marker="D", s=46,
                   facecolors=[cmap(norm(last.frame))], edgecolors="white", linewidths=1.1, zorder=7)
        # Keep labels directed towards the interior near either horizontal edge.
        first_right = first.center_x < width / 2
        last_right = last.center_x < width * .80
        if len(points) > 1:
            ax.annotate(f"ID {ident} / F{int(first.frame)}", (first.center_x, first.center_y),
                        xytext=(5 if first_right else -5, -15), textcoords="offset points",
                        ha="left" if first_right else "right", color="white", fontsize=9,
                        path_effects=text_halo, zorder=9)
        ax.annotate(f"ID {ident} / F{int(last.frame)}", (last.center_x, last.center_y),
                    xytext=(8 if last_right else -8, 7), textcoords="offset points",
                    ha="left" if last_right else "right", color="white", fontsize=9,
                    path_effects=text_halo, zorder=9)
    fig.text(.085, .956, "Particle paths + time-colored circle trails", fontsize=18,
             weight="bold", color="#192733")
    fig.text(.085, .920, f"Frames {frame_start}-{frame_end} | {len(track_ids)} selected IDs"
             f" | ROI circles every {circle_interval} frame(s), plus first / last observed",
             fontsize=10, color="#5a6877")
    cax = fig.add_axes([.845, .335, .023, .47])
    ticks = np.unique(np.linspace(frame_start, frame_end, min(6, frame_end-frame_start+1), dtype=int))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=ticks)
    colorbar.ax.invert_yaxis()
    colorbar.set_label("Frame index (early to late)", labelpad=12)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, length=0, pad=7)
    handles = [Line2D([], [], marker="o", color="#364252", markerfacecolor="none", linestyle="none", label="First observed"),
               Line2D([], [], marker="D", color="#364252", linestyle="none", label="Last observed"),
               Line2D([], [], marker=".", color="#364252", linestyle="none", label="Every observed center")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.075, .158), ncol=3,
               frameon=False, fontsize=10, handletextpad=.5, columnspacing=1.3)
    fig.text(.09, .132, "Circles show recorded ROI radius, not segmented particle boundaries.", fontsize=10, color="#334658")
    fig.text(.09, .105, "Gaps and manual corrections break paths. No interpolation or drift correction.", fontsize=9, color="#526070")
    fig.text(.09, .078, f"Background: frame {frame_start}. Raw pixels / frame indices; experimental calibration is not applied.",
             fontsize=9, color="#526070")
    missing = metadata["missing_ids"]
    if missing:
        label = "No valid observation in range: " + ", ".join(map(str, missing))
    else:
        label = f"{metadata['observed_count']} observed centers | {metadata['circle_count']} ROI circles. Colors indicate time, IDs indicate identity."
    fig.text(.09, .046, label, fontsize=9, color="#657383")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, format="png", dpi=180, facecolor=fig.get_facecolor())
    finally:
        fig.clear()
    return metadata
