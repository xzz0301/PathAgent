"""Coordinate and event persistence checks without model loading."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from trajectory_viz import TraceRecorder, export_trajectory_artifacts, split_zoom_patch


class TrajectoryTests(unittest.TestCase):
    def test_zoom_preserves_coordinate_units_and_native_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "100_200.png"
            Image.new("RGB", (512, 512), "pink").save(path)
            parts = split_zoom_patch(path, 20, 10, 4)
            self.assertEqual([origin for _, origin in parts],
                             [(100, 200), (1124, 200), (100, 1224), (1124, 1224)])
            self.assertEqual([img.size for img, _ in parts], [(256, 256)] * 4)
            with self.assertRaises(ValueError):
                split_zoom_patch(path, 15, 10, 4)

    def test_unfinished_observation_is_durable_but_not_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = TraceRecorder(folder, "run")
            started = trace.start_observation(patch_name="100_200.png", round=1)
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual(events[-1]["status"], "started")
            trace.complete_observation(started, "example description")
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["observation_started", "observation_completed"])
            self.assertEqual(trace.points[0]["status"], "completed")
            self.assertGreaterEqual(trace.points[0]["inference_ms"], 0)

    def test_center_export_uses_scale_and_retains_geometry_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.new("RGB", (512, 512), "pink").save(Path(folder) / "100_200.png")
            geometry = {"coordinate_scale": 4, "coordinate_space": "sicap_filename_inferred", "level0_verified": False}
            point = {"step": 1, "round": 1, "action": "initial_observation", "patch_name": "100_200.png",
                     "x": 100, "y": 200, "width": 512, "height": 512, "magnification": 10}
            paths = export_trajectory_artifacts([point], folder, folder, "case", "run", geometry)
            with open(paths["csv"], encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual((float(row["x_px"]), float(row["y_px"])), (1124, 1224))
            self.assertEqual(row["level0_verified"], "False")
            saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["geometry"], geometry)
            with Image.open(paths["image"]) as image:
                self.assertEqual(image.size, (1320, 930))


if __name__ == "__main__":
    unittest.main()
