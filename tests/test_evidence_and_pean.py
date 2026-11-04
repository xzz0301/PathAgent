import json
import struct
import tempfile
import unittest
from pathlib import Path

from models.grounding import parse_evidence,parse_pixel_evidence,map_regions
from experts.pean import read_epr,project_frame,spatial_visits,tissue_mask
from trajectory import TraceRecorder
from PIL import Image


class GroundingTests(unittest.TestCase):
    def test_explicit_pixel_protocol_maps_declared_canvas(self):
        payload={"coordinate_space":"input_pixels","image_size":[504,252],"regions":[{"id":"E1","bbox":[126,63,252,126],"finding":"structure"}],"primary_id":"E1"}
        self.assertEqual(parse_pixel_evidence(json.dumps(payload),(504,252))["regions"][0]["bbox_norm"],[.25,.25,.5,.5])
        payload['image_size']=[512,256]
        with self.assertRaisesRegex(ValueError,'matching canvas'):parse_pixel_evidence(json.dumps(payload),(504,252))
        payload['image_size']=[504,252];payload['regions'][0]['bbox']=[0,0,505,252]
        with self.assertRaisesRegex(ValueError,'Out-of-range'):parse_pixel_evidence(json.dumps(payload),(504,252))

    def test_affine_mapping_uses_original_roi_not_model_resize(self):
        e=parse_evidence(json.dumps({"regions":[{"id":"E1","bbox":[.1,.2,.3,.4],"finding":"test morphology"}],"primary_id":"E1"}))
        r=map_regions(e,{"x0":1000,"y0":2000,"width0":2048,"height0":1024})["regions"][0]
        self.assertEqual(r["bbox_level0"],[1204.8,2204.8,1614.4,2409.6])
        self.assertTrue(r["is_primary"])

    def test_out_of_range_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError,"Out-of-range"):
            parse_evidence('{"regions":[{"id":"E1","bbox":[0,0,100,100],"finding":"x"}],"primary_id":"E1"}')

    def test_no_evidence_and_malformed_remain_distinct(self):
        self.assertEqual(parse_evidence('{"regions":[],"primary_id":null,"no_evidence_reason":"Insufficient detail"}')["regions"],[])
        with self.assertRaises(ValueError):parse_evidence('{"regions":[')

    def test_capture_keeps_exact_observed_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=TraceRecorder(folder,"case")
            image=Image.new("RGB",(11,7),(123,45,67))
            row,_=recorder.start_observation(image=image,x=0,y=0)
            with Image.open(row["input_image_path"]) as saved:
                self.assertEqual(saved.tobytes(),image.tobytes())


class EPRTests(unittest.TestCase):
    def test_coloured_slide_background_is_not_tissue(self):
        image=Image.new('RGB',(40,40),(134,136,115))
        self.assertFalse(tissue_mask(image).any())
        image.paste((170,80,150),(10,10,30,30))
        mask=tissue_mask(image)
        self.assertTrue(mask[20,20])
        self.assertFalse(mask[5,5])

    def fixture(self,version=21):
        def s(value):return bytes([len(value)])+value.encode()
        return (b'EPRFILE'+struct.pack('<iii',version,50,2)+b'DOCTOR'+s('masked')+bytes(9)+
                b'SLIDE'+s('slide.ndpi')+s('hash')+struct.pack('<qqqqi',1000,2000,1000,2000,4)+
                b'SCREEN'+struct.pack('<iiff',1920,1080,50,30)+bytes(24)+b'DATA'+
                struct.pack('<iiiiidd',-100,-50,5,200,300,600,12.0)+
                struct.pack('<iiiiidd',-100,-50,5,210,300,600,12.166545))

    def test_versions_and_no_invented_millisecond_time(self):
        with tempfile.TemporaryDirectory() as folder:
            for version in (21,22):
                p=Path(folder)/'sample.epr';p.write_bytes(self.fixture(version))
                r=read_epr(p)
                self.assertEqual(r['version'],version)
                self.assertEqual(r['time_status'],'uncalibrated')
                frame=project_frame(r['frames'][0],r,[1,2,4])
                self.assertEqual((frame['gaze_x_level0'],frame['gaze_y_level0']),(600,700))
                self.assertIsNone(frame['time_ms'])

    def test_truncation_is_bounded_error(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'broken.epr';p.write_bytes(self.fixture()[:-3])
            with self.assertRaisesRegex(ValueError,'Truncated'):read_epr(p)

    def test_invalid_gap_keeps_visit_without_inventing_support(self):
        rows=[{'frame_index':i,'exclusion_reason':'invalid' if i==1 else '',
               'gaze_x_level0':100,'gaze_y_level0':100,'wsi_level':2} for i in range(3)]
        regions,visits=spatial_visits(rows,(.25,.25))
        self.assertEqual(len(regions),1)
        self.assertEqual(len(visits),1)
        self.assertTrue(all(v['valid_support_ms'] is None for v in visits))


if __name__=='__main__':unittest.main()
