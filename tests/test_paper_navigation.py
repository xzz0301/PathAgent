"""Verify the changed decision/observation contract without model weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from navigation import run_paper_case, sample_count, valid_region_plans
from trajectory import TraceRecorder


class Reader:
    native = 40

    def read(self, roi, mag):
        return Image.new('RGB', (40, 40), 'pink'), {'roi': roi, 'image_source': 'test_wsi'}


class Navigator:
    def encode_text(self, texts, **kwargs):
        return np.array([[1., 0.] for _ in texts])

    def encode_images(self, images, **kwargs):
        return np.array([[1., 0.] for _ in images])


class PaperNavigationTests(unittest.TestCase):
    def test_ceil_and_plan_identity(self):
        self.assertEqual((sample_count(64, .1), sample_count(64, .05)), (7, 4))
        regions = {'a': {'magnification': 5}}
        valid, invalid = valid_region_plans({'missing_info': 'nuclear morphology', 'region_plans': [
            {'patch_name': 'a', 'zoom_recommendation': 'Yes', 'recommended_zoom_level': 20},
            {'patch_name': 'invented', 'zoom_recommendation': 'Yes', 'recommended_zoom_level': 40}]}, regions, 40)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(valid[0]['query'], 'nuclear morphology')

    def run_flow(self, decisions, rounds):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name); folder = root / 'patches' / 'CASE'; folder.mkdir(parents=True)
        names = ['0_0.png', '320_0.png']
        for name in names:
            Image.new('RGB', (40, 40), 'pink').save(folder / name)
        args = SimpleNamespace(max_attempts=rounds, perceptor_max_new_tokens=37, patch_root=str(folder.parent))
        nav = Navigator(); recorder = TraceRecorder(root, 'case')
        with patch('navigation.patho_r1_describe', return_value='observed morphology') as describe, \
             patch('navigation.evaluate_with_llm_chain', side_effect=decisions) as evaluate, \
             patch('navigation.summarize_patches_in_chunks', return_value='fixed evidence'), \
             patch('navigation.slide_llm_answer', return_value={'answer': 'A', 'explanation': 'evidence'}), \
             patch('navigation.runtime_metadata', return_value={}), \
             patch.object(nav, 'encode_text', wraps=nav.encode_text) as query:
            result = run_paper_case(args=args, case_id='CASE', result_id='case', question='Question?', choices=['A','B'],
                descriptions=dict.fromkeys(names, 'generic'), features=dict.fromkeys(names, np.array([1., 0.])),
                initial_names=names, geometry={'coordinate_scale':8}, reader=Reader(), recorder=recorder,
                executor=None, tokenizer=None, navigator=nav, perceptor=None, processor=None)
        return result, describe, evaluate, query

    def test_two_regional_scales_then_reassessment(self):
        result, describe, evaluate, query = self.run_flow([
            {'sufficient':'No','missing_info':'invasion','region_plans':[
                {'patch_name':'0_0.png','zoom_recommendation':'Yes','recommended_zoom_level':10,'missing_info':'mitoses'},
                {'patch_name':'320_0.png','zoom_recommendation':'Yes','recommended_zoom_level':20,'missing_info':'nuclear atypia'}]},
            {'sufficient':'Yes'}], 3)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual([r['magnification'] for r in result['trajectory']], [5,5,10,20])
        self.assertEqual([c.args[0][0] for c in query.call_args_list], ['mitoses','nuclear atypia'])
        self.assertIn('20x', evaluate.call_args.args[2])
        self.assertEqual(result['termination_reason'], 'evidence_sufficient')
        self.assertTrue(all(c.kwargs['choices'] is None and c.kwargs['prompt_mode']=='paper_morphology'
                            for c in describe.call_args_list))
        for row in result['trajectory'][2:]:
            self.assertLessEqual(row['roi']['width0'],160)

    def test_last_round_does_not_add_unobserved_evidence(self):
        result, _, evaluate, query = self.run_flow([{'sufficient':'No','missing_info':'detail'}], 1)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(query.call_count, 0)
        self.assertEqual(result['termination_reason'], 'max_evaluation_rounds')
        self.assertEqual(len(result['final_evidence_ids']), 2)


if __name__ == '__main__':
    unittest.main()
