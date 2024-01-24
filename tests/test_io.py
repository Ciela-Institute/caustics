from pathlib import Path
import tempfile
import struct
import json
import torch
from safetensors.torch import save
from caustics.io import (
    _get_safetensors_header,
    _normalize_path,
    to_file,
    from_file,
    get_safetensors_metadata,
)


def test_normalize_path():
    path_obj = Path().joinpath("path", "to", "file.txt")
    # Test with a string path
    path_str = str(path_obj)
    normalized_path = _normalize_path(path_str)
    assert normalized_path == path_obj.absolute()
    assert str(normalized_path) == str(path_obj.absolute())

    # Test with a Path object
    normalized_path = _normalize_path(path_obj)
    assert normalized_path == path_obj.absolute()


def test_to_and_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "test.txt"
        data = "test data"

        # Test to file
        ffile = to_file(fpath, data)

        assert Path(ffile).exists()
        assert ffile == str(fpath.absolute())
        assert Path(ffile).read_text() == data

        # Test from file
        assert from_file(fpath) == data.encode("utf-8")


def test_get_safetensors_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "test.st"
        meta_dict = {"meta": "data"}
        tensors_bytes = save({"test1": torch.as_tensor(1.0)}, metadata=meta_dict)
        fpath.write_bytes(tensors_bytes)

        # Manually get header
        first_bytes_length = 8
        (length_of_header,) = struct.unpack("<Q", tensors_bytes[:first_bytes_length])
        expected_header = json.loads(
            tensors_bytes[first_bytes_length : first_bytes_length + length_of_header]
        )

        # Test for get header only
        assert _get_safetensors_header(fpath) == expected_header

        # Test for get metadata only
        assert get_safetensors_metadata(fpath) == meta_dict
