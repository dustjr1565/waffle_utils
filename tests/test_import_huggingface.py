import subprocess
from pathlib import Path

import pytest

from waffle_utils.file.io import remove_directory
from waffle_utils.image import DEFAULT_IMAGE_EXTENSION
from waffle_utils.video import DEFAULT_VIDEO_EXTENSION


def run(cmd):
    # Run a subprocess command with check=True
    subprocess.run(cmd.split(), check=True)


@pytest.fixture(scope="module")
def tmp_path_module():
    # Create a temporary directory
    tmp_path = Path("tmp_tests")

    # Yield the temporary directory to the test function
    yield tmp_path

    # Clean up the temporary directory after the test run
    remove_directory(tmp_path)


def zip_file(tmp_path_module, task):
    if task == "classification":
        return tmp_path_module / "mnist_huggingface_classification.zip"
    elif task == "object_detection":
        return tmp_path_module / "mnist_huggingface_detection.zip"


# Define fixtures for temporary paths and variables
def extract_dir(tmp_path_module, task):
    return tmp_path_module / "tmp/extract" / task


def data_root_dir(tmp_path_module):
    return tmp_path_module / "tmp/dataset"


def dataset_name(task):
    if task == "classification":
        return "mnist_cls"
    elif task == "object_detection":
        return "mnist_det"


# Define tests for dataset-related functions
@pytest.mark.parametrize("task", ["classification", "object_detection"])
def test_get_file_from_url(tmp_path_module, task):
    if task == "classification":
        url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_huggingface_classification.zip"
    elif task == "object_detection":
        url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist_huggingface_detection.zip"

    run(
        f"python -m waffle_utils.run get_file_from_url --url {url} --file-path {zip_file(tmp_path_module, task)}"
    )


@pytest.mark.parametrize("task", ["classification", "object_detection"])
def test_unzip(tmp_path_module, task):
    run(
        f"python -m waffle_utils.run unzip --file-path {zip_file(tmp_path_module, task)} --output-dir {extract_dir(tmp_path_module, task)}"
    )


@pytest.mark.parametrize("task", ["classification", "object_detection"])
def test_from_huggingface(tmp_path_module, task):
    run(
        f"python -m waffle_utils.run from_huggingface --dataset-dir {extract_dir(tmp_path_module, task)} --name {dataset_name(task)} --task {task}"
    )
