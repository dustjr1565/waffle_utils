import subprocess
from pathlib import Path

import py
import pytest

from waffle_utils.image import DEFAULT_IMAGE_EXTENSION
from waffle_utils.video import DEFAULT_VIDEO_EXTENSION


@pytest.fixture(scope="module")
def tmp_dir(tmpdir_factory: pytest.TempdirFactory) -> py.path.local:
    return tmpdir_factory.mktemp("tmp_tests")


def run(cmd) -> subprocess.CompletedProcess:
    # Run a subprocess command with check=True
    cmd = cmd.replace("\\", "")
    return subprocess.run(cmd.split(), check=True)


# Define tests for dataset-related functions
def test_get_file_from_url(tmp_dir: Path):
    zip_file = tmp_dir / "mnist.zip"
    result = run(
        f"python -m waffle_utils.run get_file_from_url \
            --url https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip \
            --file-path {zip_file}"
    )
    assert result.returncode == 0
    assert zip_file.exists()


def test_unzip(tmp_dir: Path):
    zip_file = tmp_dir / "mnist.zip"
    extract_dir = tmp_dir / "mnist_coco"

    result = run(
        f"python -m waffle_utils.run unzip \
            --file-path {zip_file} \
            --output-dir {extract_dir}"
    )
    assert result.returncode == 0
    assert extract_dir.exists()


def test_from_coco(tmp_dir: Path):
    dataset_name = "mnist_waffle"
    coco_file = tmp_dir / "mnist_coco" / "coco.json"
    coco_root_dir = tmp_dir / "mnist_coco" / "images"
    data_root_dir = tmp_dir
    result = run(
        f"python -m waffle_utils.run from_coco \
            --name {dataset_name} \
            --coco-file {coco_file} \
            --coco-root-dir {coco_root_dir} \
            --root-dir {data_root_dir}"
    )
    assert result.returncode == 0
    assert Path(tmp_dir / dataset_name).exists()


def export_huggingface(tmp_dir: Path):
    from waffle_utils.dataset.dataset import Dataset

    dataset = Dataset.load(name="mnist_waffle", root_dir=tmp_dir)
    dataset.split(0.8)
    dataset.export("huggingface_detection")
    dataset.export("huggingface_classification")


@pytest.mark.parametrize("task", ["classification", "object_detection"])
def test_from_huggingface(tmp_dir: Path, task: str):
    export_huggingface(tmp_dir)

    dataset_name = f"mnist_hugging_{task}"
    if task == "classification":
        dataset_dir = (
            tmp_dir / "mnist_waffle" / "exports" / "HUGGINGFACE_CLASSIFICATION"
        )
    elif task == "object_detection":
        dataset_dir = (
            tmp_dir / "mnist_waffle" / "exports" / "HUGGINGFACE_DETECTION"
        )
    data_root_dir = tmp_dir

    result = run(
        f"python -m waffle_utils.run from_huggingface \
            --name {dataset_name} \
            --task {task} \
            --root-dir {data_root_dir} \
            --dataset-dir {dataset_dir}"
    )

    assert result.returncode == 0
    assert Path(tmp_dir / dataset_name).exists()


def test_split(tmp_dir: Path):
    dataset_name = "mnist_waffle"
    data_root_dir = tmp_dir

    result = run(
        f"python -m waffle_utils.run split --name {dataset_name} --root-dir {data_root_dir} --train-split-ratio 0.8"
    )

    assert result.returncode == 0
    assert Path(data_root_dir / dataset_name / "sets").exists()


def test_export(tmp_dir: Path):
    dataset_name = "mnist_waffle"
    data_root_dir = tmp_dir

    result = run(
        f"python -m waffle_utils.run export --name {dataset_name} --root-dir {data_root_dir} --export-format yolo_detection"
    )
    assert result.returncode == 0

    result = run(
        f"python -m waffle_utils.run export --name {dataset_name} --root-dir {data_root_dir} --export-format coco_detection"
    )
    assert result.returncode == 0

    result = run(
        f"python -m waffle_utils.run export --name {dataset_name} --root-dir {data_root_dir} --export-format huggingface_detection"
    )
    assert result.returncode == 0


# Define fixtures and tests for video-related functions
@pytest.fixture(scope="module")
def test_video_path() -> Path:
    return Path("tests/video/test.mp4")


# Define tests for video-related functions
def test_extract_frames(tmp_dir: Path, test_video_path: Path):
    input_path = test_video_path
    output_dir = tmp_dir / "test_frames"
    result = run(
        f"python -m waffle_utils.run extract_frames --input-path {input_path} --output-dir {output_dir} --frame-rate 30  --output-image-extension {DEFAULT_IMAGE_EXTENSION}"
    )
    assert result.returncode == 0
    assert output_dir.exists()


def test_create_video(tmp_dir: Path):
    input_dir = tmp_dir / "test_frames"
    output_path = tmp_dir / f"test.{DEFAULT_VIDEO_EXTENSION}"
    result = run(
        f"python -m waffle_utils.run create_video --input-dir {input_dir} --output-path {output_path} --frame-rate 30"
    )
    assert result.returncode == 0
    assert output_path.exists()
