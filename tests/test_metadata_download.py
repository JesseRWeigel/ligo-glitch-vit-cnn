"""Exercise the Zenodo downloader with tiny files and mocked HTTP."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "01_download_metadata.py"
SPEC = importlib.util.spec_from_file_location("metadata_download", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class MetadataDownloadTest(unittest.TestCase):
    def setUp(self):
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.output / "H1_O3a.csv"
        self.good = b"header\n1\n"
        self.bad = b"header\n2\n"
        digest = hashlib.md5(self.good).hexdigest()
        self.meta = {
            "key": self.path.name,
            "links": {"self": "https://example.invalid/file"},
            "size": len(self.good),
            "checksum": f"md5:{digest}",
        }
        self.get = self.enterContext(patch.object(download.requests, "get"))
        self.response = MagicMock()
        self.get.return_value.__enter__.return_value = self.response
        self.response.iter_content.return_value = [self.good[:4], self.good[4:]]

    def test_verified_cache_does_not_use_network(self):
        self.path.write_bytes(self.good)
        self.assertEqual(download.download_file(self.meta, self.output), self.path)
        self.get.assert_not_called()

    def test_same_size_corrupt_cache_is_replaced(self):
        self.path.write_bytes(self.bad)
        self.assertEqual(download.download_file(self.meta, self.output), self.path)
        self.assertEqual(self.path.read_bytes(), self.good)
        self.get.assert_called_once()
        self.assertEqual(list(self.output.iterdir()), [self.path])

    def test_invalid_new_download_keeps_existing_and_leaves_no_temp_file(self):
        self.path.write_bytes(self.bad)
        replacement = b"header\n3\n"
        self.response.iter_content.return_value = [replacement]
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            download.download_file(self.meta, self.output)
        self.assertEqual(self.path.read_bytes(), self.bad)
        self.assertEqual(list(self.output.iterdir()), [self.path])

    def test_truncated_download_is_not_published(self):
        self.response.iter_content.return_value = [self.good[:4]]
        with self.assertRaisesRegex(ValueError, "Size mismatch"):
            download.download_file(self.meta, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_interrupted_stream_is_not_published(self):
        def interrupted():
            yield self.good[:4]
            raise download.requests.ConnectionError("stream interrupted")

        self.response.iter_content.return_value = interrupted()
        with self.assertRaisesRegex(download.requests.ConnectionError, "stream interrupted"):
            download.download_file(self.meta, self.output)
        self.assertEqual(list(self.output.iterdir()), [])
