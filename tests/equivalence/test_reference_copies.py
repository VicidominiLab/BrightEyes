import hashlib
from pathlib import Path

REFERENCE_DIR = Path(__file__).parent / "reference"


def test_copies_match_their_commits():
    """Keep each copied reference file byte-identical to its commit."""
    lines = (REFERENCE_DIR / "MANIFEST.txt").read_text().splitlines()
    assert lines
    for line in lines:
        info, name = line.split("\t")
        expected = info.split()[2]
        data = (REFERENCE_DIR / name).read_bytes()
        blob = b"blob %d\0" % len(data) + data
        assert hashlib.sha1(blob).hexdigest() == expected, name
