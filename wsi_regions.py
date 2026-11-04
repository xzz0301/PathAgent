"""Verified level-0 ROI geometry and direct WSI pixel acquisition."""
import math
from pathlib import Path
from PIL import Image


def observation_roi(row, geometry):
    if 'roi' in row:
        roi=dict(row['roi'])
    else:
        if geometry.get('image_backend')=='wsi':
            raise ValueError('WSI observations require per-observation geometry')
        d=float(geometry['coordinate_scale'])
        roi=dict(x0=float(row['x']),y0=float(row['y']),width0=row['width']*d,height0=row['height']*d)
    if not all(math.isfinite(roi[k]) for k in ('x0','y0','width0','height0')) or min(roi['width0'],roi['height0'])<=0:
        raise ValueError('Invalid observation footprint')
    return roi


class WSIReader:
    def __init__(self, manifest, slide=None):
        if not manifest.get('quickhash') or not manifest.get('dimensions'):
            raise ValueError('WSI manifest requires quickhash and dimensions')
        if slide is None:
            import openslide
            slide=openslide.OpenSlide(str(manifest['path']))
        self.slide=slide;self.manifest=dict(manifest)
        if list(slide.dimensions)!=manifest['dimensions'] or slide.properties.get('openslide.quickhash-1')!=manifest['quickhash']:
            slide.close();raise ValueError('WSI identity mismatch')
        self.native=float(manifest['nominal_native_magnification'])
        self.mpp=[float(slide.properties.get('openslide.mpp-'+axis,0)) or None for axis in ('x','y')]

    def close(self):self.slide.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def read(self, roi, magnification):
        values=[float(roi[k]) for k in ('x0','y0','width0','height0')]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite ROI')
        x,y,w,h=values
        if min(w,h)<=0 or x<0 or y<0 or x+w>self.slide.dimensions[0] or y+h>self.slide.dimensions[1]:
            raise ValueError('Requested ROI exceeds WSI; no silent crop fallback')
        d=self.native/float(magnification)
        if not math.isfinite(d) or d<1:raise ValueError('Requested magnification exceeds native resolution')
        # Quantize once in level-0. Downstream mapping uses this actual rectangle.
        left,top=math.floor(x),math.floor(y);right,bottom=math.ceil(x+w),math.ceil(y+h)
        w0,h0=right-left,bottom-top
        level=max(i for i,s in enumerate(self.slide.level_downsamples) if s<=d+1e-9)
        actual=float(self.slide.level_downsamples[level])
        read_size=(math.ceil(w0/actual),math.ceil(h0/actual))
        output_size=(max(1,round(w0/d)),max(1,round(h0/d)))
        if max(read_size)>16384 or read_size[0]*read_size[1]>64_000_000:
            raise ValueError('ROI read exceeds bounded pixel budget')
        rgba=self.slide.read_region((left,top),level,read_size).convert('RGBA')
        image=Image.new('RGBA',rgba.size,'white');image.alpha_composite(rgba);image=image.convert('RGB')
        extent=(0,0,w0/actual,h0/actual)
        resampled=output_size!=read_size or extent[2:]!=read_size
        if resampled:image=image.transform(output_size,Image.Transform.EXTENT,extent,Image.Resampling.BICUBIC)
        meta={'image_source':'original_wsi','wsi_path':str(self.manifest['path']),
              'wsi_quickhash':self.manifest['quickhash'],'slide_id':self.manifest.get('slide_id'),
              'requested_roi':dict(roi),'roi':dict(x0=left,y0=top,width0=w0,height0=h0),
              'requested_magnification':float(magnification),'nominal_magnification':float(magnification),
              'read_level':level,'actual_downsample':actual,'requested_downsample':d,
              'read_size':list(read_size),'output_size':list(output_size),'resampled':resampled,
              'requested_mpp':[v*d if v else None for v in self.mpp],
              'achieved_mpp':[self.mpp[0]*w0/output_size[0] if self.mpp[0] else None,
                              self.mpp[1]*h0/output_size[1] if self.mpp[1] else None],
              'output_scale_xy':[w0/output_size[0],h0/output_size[1]],
              'colour_handling':'scanner RGB; transparent pixels composited on white; no ICC conversion'}
        return image,meta
