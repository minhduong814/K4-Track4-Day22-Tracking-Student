"""Kiểm tra chuẩn bị và khởi chạy chấm bằng dữ liệu giả."""

from pathlib import Path

from evaluate_practice import stage, run_trackeval


def test_stage_uses_requested_split(tmp_path: Path) -> None:
    lab = tmp_path / "lab"
    src = lab / "video_1"
    (src / "gt").mkdir(parents=True)
    (src / "gt" / "gt.txt").write_text("nhãn giả", encoding="utf-8")
    (src / "seqinfo.ini").write_text("cấu hình giả", encoding="utf-8")
    submission = tmp_path / "video_1.txt"
    submission.write_text("kết quả giả", encoding="utf-8")
    root = tmp_path / "eval"
    stage(root, lab, submission, "dk", "LAB", "test")
    assert (root / "data/gt/mot_challenge/LAB-test/video_1/gt/gt.txt").read_text(encoding="utf-8") == "nhãn giả"
    assert (root / "data/trackers/mot_challenge/LAB-test/dk/data/video_1.txt").read_text(encoding="utf-8") == "kết quả giả"


def test_numpy_patch_runs_inside_child_process(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "run_mot_challenge.py"
    script.write_text(
        "import numpy as np, sys\n"
        "assert np.float is float\n"
        "assert np.int is int\n"
        "assert sys.argv[sys.argv.index('--SEQ_INFO') + 1] == 'video_1'\n"
        "assert sys.argv[sys.argv.index('--SPLIT_TO_EVAL') + 1] == 'train'\n"
    )
    run_trackeval(tmp_path, "dk", "LAB", "train")
