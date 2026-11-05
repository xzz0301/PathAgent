import unittest
from tools.evaluate_wsi_vqa import compatible_correct, norm, collect_rows


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


if __name__=='__main__':unittest.main()
