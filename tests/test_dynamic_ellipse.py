import unittest
from experts.observations import _matches
from tools.pean_dynamic_ellipse import DynamicConfig

class DynamicEllipseTest(unittest.TestCase):
    def test_full_axes_and_magnification(self):
        cfg=DynamicConfig(ellipse_width_px=200,ellipse_height_px=150)
        anchor=dict(x_px=1000,y_px=1000,viewer_relative_zoom=.5)
        region={'anchor':anchor,'detail_reference_box':[0,0,100000,100000]}
        def hit(dx,z):
            return _matches(dict(x_px=1000+dx,y_px=1000,viewer_relative_zoom=z),region,[0,0,100000,100000],True,cfg,False)
        self.assertTrue(hit(199,.5))
        self.assertFalse(hit(201,.5))
        self.assertFalse(hit(101,1.))
        self.assertFalse(hit(201,.25))

if __name__=='__main__':unittest.main()
