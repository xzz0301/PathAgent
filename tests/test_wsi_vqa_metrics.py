import unittest
from tools.evaluate_wsi_vqa import compatible_correct, norm


class WsiVqaMetricTests(unittest.TestCase):
    def test_missing_answer_is_incorrect_even_when_similarity_ties(self):
        self.assertFalse(compatible_correct('', 'Malignant', ['Benign','Malignant']))

    def test_exact_other_option_is_incorrect(self):
        self.assertFalse(compatible_correct('Benign','Malignant',['Benign','Malignant']))
        self.assertTrue(compatible_correct('Malignant','Malignant',['Benign','Malignant']))

    def test_strict_comparison_normalizes_only_display_variants(self):
        self.assertEqual(norm('  INVASIVE   CARCINOMA. '),'invasive carcinoma')


if __name__=='__main__':unittest.main()
