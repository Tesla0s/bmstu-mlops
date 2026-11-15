"""Regression checks that repeated runs cannot mix or erase previous evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import lab

class OutputIsolation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.results=self.root/"results";self.results.mkdir()
        self.a=patch.object(lab,"RESULTS",self.results);self.a.start()
        self.b=patch.object(lab,"WORK",self.root/"workspace");self.b.start()

    def tearDown(self):
        self.a.stop();self.b.stop();self.temp.cleanup()

    def test_repeated_preparation_keeps_every_previous_run(self):
        source=self.results/"sample";source.mkdir()
        (source/"requests.jsonl").write_text("first capture\n")
        first=lab.prepare_outputs("sample")
        self.assertFalse(source.exists())
        self.assertEqual((first/"sample/requests.jsonl").read_text(),"first capture\n")
        source.mkdir();(source/"requests.jsonl").write_text("second capture\n")
        second=lab.prepare_outputs("sample")
        self.assertNotEqual(first,second)
        self.assertEqual((first/"sample/requests.jsonl").read_text(),"first capture\n")
        self.assertEqual((second/"sample/requests.jsonl").read_text(),"second capture\n")

    def test_collector_refuses_to_append_to_an_existing_capture(self):
        source=self.results/"sample";source.mkdir()
        capture=source/"timeline.jsonl";capture.write_text('{"old":true}\n')
        with patch.object(lab,"get",side_effect=AssertionError("must not query Kubernetes")):
            with self.assertRaises(FileExistsError):
                with lab.Collector(source):pass
        self.assertEqual(capture.read_text(),'{"old":true}\n')

    def test_load_refuses_to_overwrite_previous_requests(self):
        source=self.results/"sample";source.mkdir()
        capture=source/"requests.jsonl";capture.write_text('{"old":true}\n')
        with patch.object(lab,"k",side_effect=AssertionError("must not send load")):
            with self.assertRaises(FileExistsError):
                lab.load("sample",[],directory=source)
        self.assertEqual(capture.read_text(),'{"old":true}\n')

if __name__=="__main__":
    unittest.main(verbosity=2)
