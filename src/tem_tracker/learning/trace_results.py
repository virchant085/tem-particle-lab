"""Time-trace exports from completed runs, matched to the registered video."""
from __future__ import annotations

from pathlib import Path
from threading import Lock
import uuid

import pandas as pd

from .data import AnnotationStore, atomic_json, read_json, valid_id
from ..io import sha256


_RENDER_LOCK = Lock()


def _sources(project: Path, video: dict) -> list[tuple[dict, Path]]:
    root = (project / 'results').resolve()
    candidates = [
        *sorted((root / 'learning').glob('tracking_*'), reverse=True),
        *sorted(root.glob('manual_*'), reverse=True),
        root / 'selected_particles',
    ]
    results = []
    for directory in candidates:
        directory = directory.resolve()
        if not directory.is_relative_to(root) or not (directory / 'tracks.csv').is_file():
            continue
        try:
            meta = read_json(directory / 'input_metadata.json', {})
            metrics = read_json(directory / 'metrics.json', {})
        except (OSError, ValueError):
            # An in-progress or damaged run must not hide healthy results.
            continue
        # A metrics file marks a finished run; matching file content prevents
        # drawing a different video's coordinates over the selected background.
        if (not isinstance(meta, dict) or not isinstance(metrics, dict)
                or not metrics or not meta.get('sha256') or meta['sha256'] != video['sha256']):
            continue
        if metrics.get('model_qualification') in ('synthetic_smoke', 'synthetic_workflow_test'):
            continue
        ids = metrics.get('selected_ids', [])
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 1 for x in ids):
            continue
        ids = sorted(set(ids))
        if directory.name == 'selected_particles':
            label = '样例 · 传统模板追踪（身份未经真值核验）'
        elif metrics.get('model_id'):
            label = f"模型 {metrics['model_id']} · {directory.name}"
        else:
            label = f"传统方法 {metrics.get('method', '')} · {directory.name}"
        results.append((dict(id=directory.name, label=label, track_ids=ids,
                             frame_start=0, frame_end=video['frames'] - 1), directory))
    return results


def list_trace_results(project: Path, store: AnnotationStore, video_id: str) -> dict:
    video = store.video(video_id)
    return {'results': [item for item, _ in _sources(project, video)]}


def export_time_traces(project: Path, store: AnnotationStore, payload: dict) -> dict:
    from ..time_traces import render_time_traces

    video = store.video(payload['video_id'])
    result_id = valid_id(payload['result_id'])
    found = next(((item, directory) for item, directory in _sources(project, video)
                  if item['id'] == result_id), None)
    if found is None:
        raise ValueError('找不到当前视频的已完成追踪结果，请重新选择结果来源')
    _, directory = found
    start, end = payload.get('frame_start'), payload.get('frame_end')
    if (type(start) is not int or type(end) is not int
            or not 0 <= start <= end < video['frames']):
        raise ValueError('起止帧必须是视频范围内的整数，且起始帧不能晚于结束帧')
    if sha256(store.path(video['id'])) != video['sha256']:
        raise ValueError('已导入的视频文件发生改变，请重新导入并追踪后再生成图片')
    tracks_path = directory / 'tracks.csv'
    tracks = pd.read_csv(tracks_path)
    background = store.frame(video['id'], start)
    export_dir = project / 'results/learning/time_traces'
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / ('trace_' + uuid.uuid4().hex + '.png')
    # The HTTP server is threaded; serialize Matplotlib's shared font caches.
    with _RENDER_LOCK:
        metadata = render_time_traces(background, tracks, output,
            track_ids=payload.get('track_ids'), frame_start=start, frame_end=end,
            circle_interval=payload.get('circle_interval', 5))
    metadata.update(video_id=video['id'], result_id=result_id,
                    source_tracks=tracks_path.relative_to(project).as_posix(),
                    source_tracks_sha256=sha256(tracks_path),
                    source_video_sha256=video['sha256'], background_frame=start,
                    coordinate_basis='raw_image_pixels', time_basis='frame_index')
    metadata_path = output.with_suffix('.json')
    atomic_json(metadata_path, metadata)
    return dict(metadata, image_url='/' + output.relative_to(project).as_posix(),
                metadata_url='/' + metadata_path.relative_to(project).as_posix())
