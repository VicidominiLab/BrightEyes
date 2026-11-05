"""Check packaged data and verified download behaviour."""

import hashlib
import io
from pathlib import Path

import pytest
import requests

from brighteyes import datasets


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Keep input data and downloads inside this test's directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    catalog = {}
    monkeypatch.setattr("brighteyes.datasets._fetch.data_dir", data_dir)
    monkeypatch.setattr("brighteyes.datasets._fetch.cache_dir", cache_dir)
    monkeypatch.setattr("brighteyes.datasets._fetch.catalog", catalog)
    return data_dir, cache_dir, catalog


@pytest.fixture
def download(monkeypatch):
    """Record requests and provide a real response with in-memory bytes."""
    calls = []
    response = requests.Response()
    response.status_code = 200
    response.raw = io.BytesIO(b"measurement")

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(requests, "get", get)
    return response, calls


def remote_entry(content=b"measurement"):
    """Describe a remote fixture with a known length and checksum."""
    return {
        "url": "https://example.invalid/measurement.bin",
        "filename": "measurement.bin",
        "size": len(content),
        "checksum": "sha256:" + hashlib.sha256(content).hexdigest(),
    }


def test_fetch_bundled_without_network(storage, download):
    """Return a verified packaged file without making a request."""
    data_dir, _, catalog = storage
    content = b"real sample"
    (data_dir / "sample.bin").write_bytes(content)
    catalog["sample"] = {
        "path": "sample.bin",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    path = datasets.fetch("sample")
    assert isinstance(path, Path)
    assert path.read_bytes() == content
    assert download[1] == []


def test_fetch_reuses_verified_cache(storage, download):
    """Reuse the verified download on the second fetch."""
    _, _, catalog = storage
    catalog["measurement"] = remote_entry()
    first = datasets.fetch("measurement")
    second = datasets.fetch("measurement")
    assert first == second
    assert first.read_bytes() == b"measurement"
    assert len(download[1]) == 1
    assert download[1][0][1]["stream"] is True
    assert all(value > 0 for value in download[1][0][1]["timeout"])


def test_failed_download_preserves_cache(storage, download):
    """Preserve an existing file when its replacement request fails."""
    _, _, catalog = storage
    catalog["measurement"] = remote_entry()
    path = datasets.fetch("measurement")
    path.write_bytes(b"previous file")
    response, _ = download
    response.status_code = 503
    with pytest.raises(requests.HTTPError):
        datasets.fetch("measurement")
    assert path.read_bytes() == b"previous file"
    assert list(path.parent.iterdir()) == [path]


def test_checksum_mismatch_removes_partial_file(storage, download):
    """Reject incorrect downloaded bytes without leaving a partial file."""
    _, cache_dir, catalog = storage
    catalog["measurement"] = remote_entry(b"wrong bytes")
    with pytest.raises(ValueError, match="checksum"):
        datasets.fetch("measurement")
    assert not [path for path in cache_dir.rglob("*") if path.is_file()]


def test_fetch_rejects_truncated_download(storage, download):
    """Reject a download whose length differs from the catalog."""
    _, cache_dir, catalog = storage
    catalog["measurement"] = remote_entry()
    catalog["measurement"]["size"] += 1
    with pytest.raises(ValueError, match="size"):
        datasets.fetch("measurement")
    assert not [path for path in cache_dir.rglob("*") if path.is_file()]


def test_fetch_rejects_modified_packaged_data(storage, download):
    """Report corrupted packaged input instead of returning it."""
    data_dir, _, catalog = storage
    (data_dir / "sample.bin").write_bytes(b"different")
    catalog["sample"] = {
        "path": "sample.bin",
        "size": 9,
        "sha256": hashlib.sha256(b"reference").hexdigest(),
    }
    with pytest.raises(ValueError, match="checksum"):
        datasets.fetch("sample")
    assert download[1] == []


def test_fetch_rejects_unknown_name(storage, download):
    """Reject unknown names without making a request."""
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.fetch("missing")
    assert download[1] == []
