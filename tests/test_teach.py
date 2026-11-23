import numpy as np
import pytest

from actuo import teach


def test_recorder_thins_to_its_rate_and_keeps_its_joints(tmp_path):
    rec = teach.Recorder(['a', 'b'], rate=10.0)
    for i in range(100):
        rec.add(i * 0.01, ['b', 'x', 'a'], [2.0 * i, 9.0, float(i)])
    assert len(rec.times) == 10
    assert rec.positions[1] == [10.0, 20.0]
    path = teach.save(str(tmp_path), rec.to_dict('wave_1'))
    names, times, positions = teach.load(str(tmp_path), 'wave_1')
    assert path.endswith('wave_1.yaml')
    assert names == ['a', 'b']
    assert times[0] == 0.0 and times[-1] == pytest.approx(0.9)
    assert positions.shape == (10, 2)
    assert teach.listing(str(tmp_path)) == [('wave_1', pytest.approx(0.9))]


def test_names_that_are_not_plain_file_names_are_refused(tmp_path):
    for bad in ('', '../x', 'a b', 'a/b'):
        with pytest.raises(ValueError):
            teach.check_name(bad)
    with pytest.raises(ValueError):
        teach.Recorder(['a'], 10.0).to_dict('empty')


def test_playback_timing():
    times = teach.playback_times([3.0, 3.5, 4.0], speed=0.5)
    np.testing.assert_allclose(times, [0.05, 1.05, 2.05])
    assert teach.approach_duration([0.0, 0.0], [0.1, 0.0], speed=0.4) == 1.0
    assert teach.approach_duration([0.0, 0.0], [0.0, -2.0], speed=0.4) == pytest.approx(5.0)


def test_playback_velocities_are_central_differences_and_stop_at_the_ends():
    v = teach.playback_velocities([0.0, 1.0, 2.0, 4.0], [[0.0], [1.0], [3.0], [3.0]])
    np.testing.assert_allclose(v[:, 0], [0.0, 1.5, 2.0 / 3.0, 0.0])
