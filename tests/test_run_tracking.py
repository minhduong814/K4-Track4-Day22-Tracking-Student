"""Kiểm tra truyền thiết bị bằng mô hình giả, không dùng GPU hay ảnh lab."""

from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import run_tracking


def test_run_applies_device_and_preserves_frame_numbers(tmp_path, monkeypatch) -> None:
    detector = Mock()
    tracker = Mock()
    tracker.update.return_value = np.array([[10, 20, 30, 60, 7, .9, 0]])
    factory = Mock(return_value=tracker)
    monkeypatch.setattr(run_tracking, "YOLO", Mock(return_value=detector))
    monkeypatch.setattr(run_tracking, "create_tracker", factory)
    monkeypatch.setattr(run_tracking, "iter_frames", lambda source: iter([(0, np.zeros((80, 80, 3), dtype=np.uint8)), (1, np.zeros((80, 80, 3), dtype=np.uint8))]))
    monkeypatch.setattr(run_tracking, "detect", Mock(return_value=np.empty((0, 6))))
    args = Namespace(source="nguồn giả", out=str(tmp_path), seq_name="video_1", tracker="botsort", device="cuda:0", conf=.3, iou=.5, save_video=False, max_frames=0, fps=20)
    run_tracking.run(args)
    detector.to.assert_called_once_with("cuda:0")
    assert str(factory.call_args.kwargs["device"]) == "cuda:0"
    rows = (tmp_path / "video_1.txt").read_text().splitlines()
    assert len(rows) == 2
    assert rows[0].split(",")[:6] == ["1", "7", "10.00", "20.00", "20.00", "40.00"]
    assert rows[1].startswith("2,7,")
