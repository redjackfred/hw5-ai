import os
import tempfile
import numpy as np
import pytest

from model.sgf_parser import parse_sgf_file, load_dataset


# Minimal valid 9x9 SGF with one black move at (4,4), black wins
VALID_SGF_9x9 = b"(;GM[1]FF[4]SZ[9]RE[B+1.5];B[ee])"

# 19x19 SGF (wrong size)
SGF_19x19 = b"(;GM[1]FF[4]SZ[19]RE[B+1.5];B[ee])"

# 9x9 SGF with no RE property (no winner)
SGF_NO_WINNER = b"(;GM[1]FF[4]SZ[9];B[ee])"


def write_sgf(tmp_path, content: bytes, filename: str = "test.sgf") -> str:
    path = os.path.join(tmp_path, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def test_parse_sgf_file_returns_list(tmp_path):
    path = write_sgf(str(tmp_path), VALID_SGF_9x9)
    result = parse_sgf_file(path)
    assert isinstance(result, list)


def test_parse_sgf_file_wrong_size_returns_empty(tmp_path):
    path = write_sgf(str(tmp_path), SGF_19x19)
    result = parse_sgf_file(path)
    assert result == []


def test_parse_sgf_file_no_winner_returns_empty(tmp_path):
    path = write_sgf(str(tmp_path), SGF_NO_WINNER)
    result = parse_sgf_file(path)
    assert result == []


def test_sample_shapes(tmp_path):
    path = write_sgf(str(tmp_path), VALID_SGF_9x9)
    result = parse_sgf_file(path)
    assert len(result) >= 1, "Expected at least one sample from a valid SGF"
    feat, pol, val = result[0]
    assert feat.shape == (17, 9, 9), f"Expected feat shape (17,9,9), got {feat.shape}"
    assert pol.shape == (82,), f"Expected pol shape (82,), got {pol.shape}"
    assert val in (-1.0, 1.0), f"Expected val in {{-1.0, 1.0}}, got {val}"


def test_load_dataset_no_files_raises(tmp_path):
    with pytest.raises(RuntimeError):
        load_dataset(str(tmp_path))
