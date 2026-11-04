import json,unittest
from pathlib import Path
from experts.observations import simplify,filter_observations
from experts.compression import compress
from experts.pean_aligned import align_samples

class EstablishedRuleTests(unittest.TestCase):
    def test_matches_frozen_original_pathogaze_contract(self):
        f=json.loads((Path(__file__).parent/'fixtures/established_gaze_contract.json').read_text())
        actual=filter_observations(simplify(f['events'],f['states']),150)
        for s in actual['steps']:s['action_linked']=False
        actual=compress(actual)
        self.assertEqual(actual,f['expected'])
    def rows(self,n):
        return [dict(frame_index=i,gaze_x_level0=1000,gaze_y_level0=1000,downsample=4,display_left_level0=0,
                     display_top_level0=0,display_width_level0=7680,display_height_level0=4320,valid_spatial_sample=True,exclusion_reason='') for i in range(n)]
    def test_gap_preserves_visit_but_never_adds_time(self):
        rows=self.rows(12);rows[4].update(valid_spatial_sample=False,exclusion_reason='invalid')
        result=align_samples(rows,(100000,100000),50,'fps_estimate',150,False)
        self.assertEqual(len(result['steps']),1)
        self.assertEqual(result['steps'][0]['estimated_support_ms'],200)
        self.assertIsNone(result['steps'][0]['valid_support_ms'])
    def test_single_sample_not_qualified_by_150ms_estimate(self):
        self.assertFalse(align_samples(self.rows(1),(100000,100000),50,'fps_estimate')['steps'])
        spatial=align_samples(self.rows(1),(100000,100000),50)
        self.assertEqual(len(spatial['steps']),1)
        self.assertFalse(spatial['duration_filter_applied']);self.assertIsNone(spatial['steps'][0]['valid_support'])
    def test_filter_then_contiguous_display_numbers(self):
        rows=self.rows(14)
        rows[0]['gaze_x_level0']=60000;rows[0]['display_left_level0']=55000
        result=align_samples(rows,(100000,100000),50,'fps_estimate')
        self.assertEqual([s['step'] for s in result['steps']],[1])
        self.assertEqual([r['region_id'] for r in result['regions']],[1])
        self.assertEqual(result['regions'][0]['x_px'],1000)
if __name__=='__main__':unittest.main()
