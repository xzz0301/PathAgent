"""Evaluation must not drop invalid predictions from the frozen cohort."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvaluationTest(unittest.TestCase):
    def test_invalid_and_missing_keep_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            (root / 'agent').mkdir()
            labels = []
            manifest = {}
            predictions = ['Malignant', 'Benign', 'uncertain', 'Benign', 'Malignant', None]
            for i, prediction in enumerate(predictions):
                case = 'C' + str(i + 1)
                labels.append(dict(case_id=case, long_id=case, slide_id='test_' + str(i),
                                   label='Malignant' if i < 3 else 'Benign', official={'class': 'test'}))
                manifest[case] = {'path': 'unused'}
                if prediction is not None:
                    (root / ('roi_' + case)).mkdir()
                    (root / 'agent' / (case + '.json')).write_text(json.dumps(dict(
                        long_id=case, pred_answer=prediction, ground_truth='WITHHELD', trajectory=[])))
            (root / 'inputs/evaluation_labels.json').write_text(json.dumps(labels))
            (root / 'inputs/wsi_manifest.json').write_text(json.dumps(manifest))
            script = Path(__file__).resolve().parents[1] / 'tools/evaluate_camelyon_pilot.py'
            subprocess.run([sys.executable, str(script), '--root', str(root)], check=True, capture_output=True)
            metrics = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(metrics['n'], 6)
            self.assertEqual(metrics['correct'], 2)
            self.assertEqual(metrics['accuracy'], 2 / 6)
            self.assertEqual(metrics['sensitivity'], 1 / 3)
            self.assertEqual(metrics['specificity'], 1 / 3)
            for field in ('TP', 'TN', 'FP', 'FN', 'invalid_positive', 'invalid_negative'):
                self.assertEqual(metrics[field], 1)


if __name__ == '__main__':
    unittest.main()
