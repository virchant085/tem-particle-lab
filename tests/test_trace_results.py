from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tem_tracker.learning.data import AnnotationStore, atomic_json
from tem_tracker.learning.trace_results import list_trace_results, export_time_traces
from tem_tracker.io import sha256


@pytest.fixture
def result_project(tmp_path):
    source = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((64, 64, 3), 30 + index * 40, np.uint8))
    writer.release()
    store = AnnotationStore(tmp_path)
    video = store.register(source, 'test')
    directory = tmp_path / 'results/learning/tracking_test'
    directory.mkdir(parents=True)
    pd.DataFrame([dict(frame=i, track_id=2, center_x=15+i*3, center_y=30,
                       radius_px=5, observed=True, status='seed' if i == 0 else 'tracked')
                  for i in range(4)]).to_csv(directory / 'tracks.csv', index=False)
    atomic_json(directory / 'input_metadata.json', {'sha256': video['sha256']})
    atomic_json(directory / 'metrics.json', dict(selected_ids=[2], model_id='test_model', frames=4))
    return tmp_path, store, video, directory


def payload(video, **changes):
    return dict(dict(video_id=video['id'], result_id='tracking_test', track_ids=[2],
                     frame_start=0, frame_end=3, circle_interval=2), **changes)


def test_only_completed_matching_video_results_are_listed(result_project):
    project, store, video, directory = result_project
    assert list_trace_results(project, store, video['id'])['results'][0]['track_ids'] == [2]
    atomic_json(directory / 'input_metadata.json', {'sha256': 'other-video'})
    assert list_trace_results(project, store, video['id']) == {'results': []}
    atomic_json(directory / 'input_metadata.json', {'sha256': video['sha256']})
    (directory / 'metrics.json').unlink()
    assert list_trace_results(project, store, video['id']) == {'results': []}


def test_export_preserves_measurements_and_uses_range_start_background(result_project, monkeypatch):
    project, store, video, directory = result_project
    before = sha256(directory / 'tracks.csv')
    frames = []
    original = store.frame
    def read_frame(video_id, index):
        frames.append(index)
        return original(video_id, index)
    monkeypatch.setattr(store, 'frame', read_frame)
    result = export_time_traces(project, store, payload(video, frame_start=1))
    assert frames == [1]
    assert result['background_frame'] == 1 and result['time_basis'] == 'frame_index'
    assert result['observed_count'] == 3 and result['circle_count'] == 2
    image = project / result['image_url'].lstrip('/')
    assert image.is_file() and cv2.imread(str(image)) is not None
    meta = json.loads((project / result['metadata_url'].lstrip('/')).read_text(encoding='utf-8'))
    assert meta['source_tracks_sha256'] == before == sha256(directory / 'tracks.csv')
    assert meta['source_video_sha256'] == sha256(store.path(video['id']))


def test_changed_video_is_not_used_with_old_coordinates(result_project):
    project, store, video, _ = result_project
    store.path(video['id']).write_bytes(b'replaced video')
    with pytest.raises(ValueError, match='发生改变'):
        export_time_traces(project, store, payload(video))


def test_synthetic_training_results_are_not_offered_as_tem(result_project):
    project, store, video, directory = result_project
    atomic_json(directory / 'metrics.json', dict(selected_ids=[2],
                model_qualification='synthetic_workflow_test'))
    assert list_trace_results(project, store, video['id']) == {'results': []}


def test_incomplete_result_metadata_does_not_hide_healthy_results(result_project):
    project, store, video, directory = result_project
    broken = directory.with_name('tracking_incomplete')
    broken.mkdir()
    (broken / 'tracks.csv').write_bytes((directory / 'tracks.csv').read_bytes())
    atomic_json(broken / 'input_metadata.json', {'sha256': video['sha256']})
    (broken / 'metrics.json').write_text('{"selected_ids":', encoding='utf-8')
    assert [item['id'] for item in list_trace_results(project, store, video['id'])['results']] == ['tracking_test']


@pytest.mark.parametrize('changes', [dict(frame_start=-1), dict(frame_end=4),
    dict(frame_start=2, frame_end=1), dict(frame_start=1.5), dict(frame_start=True),
    dict(result_id='../tracking_test'), dict(result_id='tracking_other'),
    dict(track_ids=[]), dict(track_ids=[999]), dict(circle_interval=0)])
def test_invalid_exports_are_rejected(result_project, changes):
    project, store, video, _ = result_project
    with pytest.raises(ValueError):
        export_time_traces(project, store, payload(video, **changes))
    assert not list((project / 'results/learning/time_traces').glob('*.png'))
