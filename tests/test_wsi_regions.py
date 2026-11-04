import unittest
from PIL import Image
from wsi_regions import WSIReader,observation_roi

class Slide:
    dimensions=(4096,4096);level_downsamples=(1.,4.)
    properties={'openslide.quickhash-1':'known','openslide.mpp-x':'.25','openslide.mpp-y':'.25'}
    def read_region(self,location,level,size):
        self.last=(location,level,size);return Image.new('RGBA',size,(120,40,130,255))
    def close(self):pass

class WSITests(unittest.TestCase):
    def reader(self):return WSIReader(dict(path='fixture',quickhash='known',dimensions=[4096,4096],nominal_native_magnification=40),Slide())
    def test_true_zoom_keeps_footprint_and_adds_source_pixels(self):
        with self.reader() as reader:
            roi=dict(x0=1024,y0=1024,width0=1024,height0=1024)
            low,l=reader.read(roi,10);high,h=reader.read(roi,20)
            self.assertEqual(low.size,(256,256));self.assertEqual(high.size,(512,512))
            self.assertEqual(l['roi'],h['roi']);self.assertEqual(h['read_level'],0)
            self.assertEqual(h['read_size'],[1024,1024]);self.assertTrue(h['resampled'])
            self.assertEqual(h['achieved_mpp'],[.5,.5])
    def test_wrong_slide_and_outside_roi_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'):WSIReader(dict(path='fixture',quickhash='wrong',dimensions=[4096,4096],nominal_native_magnification=40),Slide())
        with self.reader() as reader:
            with self.assertRaisesRegex(ValueError,'exceeds WSI'):reader.read(dict(x0=4000,y0=0,width0=200,height0=200),20)
            with self.assertRaisesRegex(ValueError,'native'):reader.read(dict(x0=0,y0=0,width0=200,height0=200),80)
    def test_per_observation_geometry_beats_legacy_global_scale(self):
        row=dict(x=10,y=20,width=512,height=512,roi=dict(x0=10,y0=20,width0=1024,height0=1024))
        self.assertEqual(observation_roi(row,{'coordinate_scale':4})['width0'],1024)
        with self.assertRaisesRegex(ValueError,'per-observation'):
            observation_roi(dict(x=0,y=0,width=512,height=512),{'coordinate_scale':4,'image_backend':'wsi'})
    def test_edge_and_fractional_request_records_actual_rectangle(self):
        with self.reader() as reader:
            image,meta=reader.read(dict(x0=3072.5,y0=3072.5,width0=1023.5,height0=1023.5),20)
            self.assertEqual(meta['roi'],dict(x0=3072,y0=3072,width0=1024,height0=1024))
            self.assertEqual(image.size,(512,512))
if __name__=='__main__':unittest.main()
