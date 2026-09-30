from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tem_tracker.time_traces import _prepare_tracks, _time_scale, render_time_traces


def row(frame, ident=1, **changes):
    result = dict(frame=frame, track_id=ident, center_x=20 + frame * 2,
                  center_y=30 + frame, radius_px=8 + frame / 10,
                  observed=True, status="tracked")
    result.update(changes)
    return result


@pytest.fixture
def background():
    return np.full((96, 128), 160, np.uint8)


def test_geometry_does_not_connect_gaps_invalid_points_or_manual_anchors():
    tracks = pd.DataFrame([
        row(0), row(1), row(2, observed=False), row(3), row(4, center_x=np.inf),
        row(5), row(6, status="manual_anchor"), row(7), row(9), row(10),
        row(11, radius_px=0), row(12), row(13, center_y=np.nan), row(14),
    ])
    particle = _prepare_tracks(tracks, [1], 0, 14, 5)[0]
    assert particle["segment_frames"] == [[0, 1], [6, 7], [9, 10]]
    assert particle["segments"] == [
        [[20., 30.], [22., 31.]], [[32., 36.], [34., 37.]], [[38., 39.], [40., 40.]],
    ]
    assert particle["segment_times"] == [.5, 6.5, 9.5]
    assert particle["points"].frame.tolist() == [0, 1, 3, 5, 6, 7, 9, 10, 12, 14]
    assert particle["samples"].frame.tolist() == [0, 5, 10, 14]


def test_sample_grid_uses_global_range_start_and_skips_missing_slots():
    tracks = pd.DataFrame([row(frame, ident=2) for frame in range(2, 14)]
                          + [row(frame) for frame in [3, 4, 6, 8, 9, 10, 11]])
    particles = _prepare_tracks(tracks, [1, 2], 2, 13, 5)
    assert particles[0]["samples"].frame.tolist() == [3, 11]  # Grid 2, 7, 12 all missing.
    assert particles[1]["samples"].frame.tolist() == [2, 7, 12, 13]
    assert particles[0]["segment_frames"] == [[3, 4], [8, 9], [9, 10], [10, 11]]
    assert particles[1]["points"].iloc[-1].radius_px == pytest.approx(9.3)


def test_export_filters_ids_and_range_preserves_input_and_notes_missing(tmp_path, background):
    tracks = pd.DataFrame([row(frame, ident) for ident in (1, 2, 3) for frame in range(7)])
    tracks.loc[(tracks.track_id == 2) & tracks.frame.between(2, 5), "observed"] = False
    original = tracks.copy(deep=True)
    output = tmp_path / "nested" / "trace.png"
    metadata = render_time_traces(background, tracks, output, track_ids=[2, 1],
                                  frame_start=2, frame_end=5, circle_interval=2)
    assert metadata["track_ids"] == [2, 1]
    assert metadata["missing_ids"] == [2]
    assert metadata["observed_count"] == 4
    assert metadata["circle_count"] == 3
    assert metadata["segment_count"] == 3
    assert metadata["background_frame"] == 2
    assert metadata["particles"][0]["first_frame"] is None
    assert metadata["particles"][1]["sampled_frames"] == [2, 4, 5]
    assert metadata["particles"][1]["first_frame"] == 2
    assert metadata["particles"][1]["last_frame"] == 5
    assert metadata["time_basis"] == "frame_index"
    json.dumps(metadata, allow_nan=False)
    pd.testing.assert_frame_equal(tracks, original)
    image = cv2.imdecode(np.fromfile(output, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (1800, 1800, 3)
    assert image.std() > 25  # Nonempty raster with visible figure content.


def test_single_frame_keeps_isolated_point_one_circle_and_constant_color(tmp_path, background):
    tracks = pd.DataFrame([row(7)])
    metadata = render_time_traces(background, tracks, tmp_path / "one.png", track_ids=[1],
                                  frame_start=7, frame_end=7)
    assert metadata["observed_count"] == metadata["circle_count"] == 1
    assert metadata["segment_count"] == 0
    assert metadata["particles"][0]["sampled_frames"] == [7]
    assert metadata["particles"][0]["first_frame"] == metadata["particles"][0]["last_frame"] == 7
    cmap, norm = _time_scale(7, 7)
    assert np.isfinite(norm(7))
    assert cmap(norm(7)) == cmap(0.) == cmap(1.)


def test_common_time_scale_does_not_stretch_shorter_tracks_to_end():
    cmap, norm = _time_scale(10, 30)
    assert norm(10) == 0
    assert norm(20) == .5
    assert norm(30) == 1
    assert cmap(norm(20)) != cmap(norm(30))


@pytest.mark.parametrize("change", [
    {"track_ids": []}, {"track_ids": [1, 1]}, {"track_ids": [9]},
    {"track_ids": [1.]}, {"track_ids": [True]},
    {"frame_start": -1}, {"frame_start": 1.5}, {"frame_start": True},
    {"frame_start": 3, "frame_end": 1}, {"frame_end": "1"},
    {"circle_interval": 0}, {"circle_interval": 2.5}, {"circle_interval": False},
])
def test_invalid_requests_fail_before_output(tmp_path, background, change):
    args = dict(track_ids=[1], frame_start=0, frame_end=2, circle_interval=1)
    args.update(change)
    output = tmp_path / "never.png"
    with pytest.raises(ValueError):
        render_time_traces(background, pd.DataFrame([row(0), row(1)]), output, **args)
    assert not output.exists()


@pytest.mark.parametrize("change", [
    {"observed": False}, {"observed": "False"}, {"observed": None},
    {"center_x": np.nan}, {"center_y": np.inf}, {"radius_px": -1},
    {"radius_px": np.inf},
])
def test_all_invalid_or_missing_observations_have_chinese_error(tmp_path, background, change):
    with pytest.raises(ValueError, match="没有有效观测"):
        render_time_traces(background, pd.DataFrame([row(0, **change)]), tmp_path / "none.png",
                           track_ids=[1], frame_start=0, frame_end=1)


def test_duplicate_positions_in_a_frame_are_rejected(tmp_path, background):
    with pytest.raises(ValueError, match="重复"):
        render_time_traces(background, pd.DataFrame([row(0), row(0, center_x=88)]),
                           tmp_path / "none.png", track_ids=[1], frame_start=0, frame_end=1)


def test_known_id_with_no_rows_in_range_is_missing(tmp_path, background):
    with pytest.raises(ValueError, match="没有有效观测"):
        render_time_traces(background, pd.DataFrame([row(0)]), tmp_path / "none.png",
                           track_ids=[1], frame_start=2, frame_end=3)
