import unittest
import json
import tempfile
from pathlib import Path
from tools.evaluate_wsi_vqa import compatible_correct, norm, collect_rows, require_eligible_coverage
from data_processing.utils import load_all_vqa_pairs


class WsiVqaMetricTests(unittest.TestCase):
    def test_missing_answer_is_incorrect_even_when_similarity_ties(self):
        self.assertFalse(compatible_correct('', 'Malignant', ['Benign','Malignant']))

    def test_exact_other_option_is_incorrect(self):
        self.assertFalse(compatible_correct('Benign','Malignant',['Benign','Malignant']))
        self.assertTrue(compatible_correct('Malignant','Malignant',['Benign','Malignant']))

    def test_strict_comparison_normalizes_only_display_variants(self):
        self.assertEqual(norm('  INVASIVE   CARCINOMA. '),'invasive carcinoma')

    def test_uncalibrated_qa_stays_in_reference_denominator(self):
        reference=[{'Id':'good','Question':'q1','Choice':['yes','no'],'Answer':'yes'},
                   {'Id':'bad','Question':'q2','Choice':['yes','no'],'Answer':'yes'},
                   {'Id':'bad','Question':'q3','Answer':'normal tissue'}]
        result={'ground_truth':'WITHHELD','pred_answer':'yes','process':[],'trajectory':[]}
        rows,closed,opened=collect_rows(reference,{'good':'SLIDE'},{('SLIDE','q1'):(result,None)},
                                        {'bad':'missing_objective_magnification'})
        self.assertEqual((len(rows),len(closed),len(opened)),(3,2,1))
        self.assertEqual(sum(r['compatible_correct'] for r in closed),1)
        self.assertEqual(sum(r['calibration_eligible'] for r in rows),1)
        self.assertEqual(rows[1]['status'],'uncalibrated')
        self.assertEqual(opened[0]['prediction'],'')

    def test_dx2_questions_are_loaded_and_unrelated_paths_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);patches=root/'patches';patches.mkdir()
            for name in ['TCGA-D8-A73U-01Z-00-DX2.scan','TCGA-D8-A1JK-01Z-00-DX1.scan',
                         'TCGA-XX-XXXX-01Z-00-DX1.unrelated']:
                (patches/name).mkdir()
            (patches/'TCGA-D8-A73U-file-not-a-folder').write_text('ignored')
            questions=[{'Id':'TCGA-D8-A73U','Question':f'question {i}','Answer':'WITHHELD'} for i in range(10)]
            questions.append({'Id':'TCGA-D8-A1JK','Question':'another','Answer':'WITHHELD'})
            source=root/'questions.json';source.write_text(json.dumps(questions))
            pairs=load_all_vqa_pairs(str(source),image_dir=str(patches))
            self.assertEqual(len(pairs),11)
            self.assertEqual(sum(p['short_id']=='TCGA-D8-A73U' for p in pairs),10)
            self.assertTrue(all(p['long_id'].endswith('.scan') for p in pairs))

    def test_incomplete_eligible_results_cannot_be_finalized(self):
        with self.assertRaisesRegex(ValueError,'1 unanswered eligible'):
            require_eligible_coverage([{'index':3,'calibration_eligible':True,'prediction':''}])
        require_eligible_coverage([{'index':3,'calibration_eligible':False,'prediction':''},
                                  {'index':4,'calibration_eligible':True,'prediction':'answer'}])


if __name__=='__main__':unittest.main()
