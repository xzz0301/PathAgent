"""Prepare a C1 candidate set using image tissue only, without expert coordinates."""
import io
import json
import math
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.fetch_camelyon_previews import fetch_range
from demo.inspect_remote_tiff import parse_value

BASE = Path(r"D:\PathCode\demo\c1_independent")
URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16/images/test_001.tif"
CASE = "DX1_C1"  # compatibility with upstream's DX1-only loader


def directories():
    class Reader:
        def read(self, offset, length):
            return fetch_range(URL, offset, offset+length-1)
    reader = Reader()
    head = reader.read(0,16)
    endian = '<' if head[:2] == b'II' else '>'
    assert struct.unpack(endian+'H', head[2:4])[0] == 43
    offset = struct.unpack(endian+'Q',head[8:])[0]
    result = []
    while offset:
        count = struct.unpack(endian+'Q',reader.read(offset,8))[0]
        entries = reader.read(offset+8,count*20+8)
        tags = {}
        for index in range(count):
            item = entries[index*20:(index+1)*20]
            tag,typ = struct.unpack(endian+'HH',item[:4])
            if tag not in (256,257,270,322,323,324,325,347): continue
            n = struct.unpack(endian+'Q',item[4:12])[0]
            value,_ = parse_value(reader,endian,typ,n,item[12:])
            tags[tag] = value
        result.append(tags)
        print('IFD',tags.get(256),tags.get(257),flush=True)
        # This slide's 1:4 image has dimensions rounded to tile boundaries.
        if tags.get(256) == 21504 and 22400 <= tags.get(257,0) < 22912:
            return result
        offset = struct.unpack(endian+'Q',entries[count*20:])[0]
    raise ValueError('C1 downsample 4 image not found')


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    cache=BASE/'tiff_directories.npy'
    levels=np.load(cache,allow_pickle=True).tolist() if cache.exists() else directories()
    desc = ''.join(levels[0].get(270,()))
    (BASE/'wsi_description.xml').write_text(desc,encoding='utf-8')
    np.save(BASE/'tiff_directories.npy', np.array(levels,dtype=object),allow_pickle=True)
    print('WSI description:',desc[:800],flush=True)
    assert levels[0][256] == 86016 and levels[0][257] == 89600
    import re
    print('Magnification/spacing metadata:',[(line.strip()) for line in desc.splitlines() if re.search('magni|spacing|objective|resolution|pixel_size',line,re.I)],flush=True)
    # Scanner metadata is Hamamatsu XML rather than Aperio AppMag.
    # XML verifies physical spacing, not an objective magnification field.
    # Display 40/ds as a nominal convention and record that limitation.
    assert '0.000227273' in desc and '0.000909091' in desc
    tags=levels[-1]
    assert tags[322]==512 and tags[323]==512
    overview=Image.open(r'D:\pathologyData\CAMELYON16\pathogaze_pilot\C1_test_001_overview.png').convert('RGB')
    hsv=np.asarray(overview.convert('HSV'));rgb=np.asarray(overview)
    tissue=(hsv[:,:,1]>35)&(rgb.min(axis=2)<220)&(rgb.max(axis=2)>50)
    eligible=[]
    for row in range(math.ceil(89600/2048)):
        for col in range(math.ceil(86016/2048)):
            mask=tissue[row*32:min((row+1)*32,tissue.shape[0]),col*32:min((col+1)*32,tissue.shape[1])]
            fraction=float(mask.mean()) if mask.size else 0
            if fraction>=0.25: eligible.append((col,row,fraction))
    selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(64,len(eligible)),dtype=int)]
    patchdir=BASE/'patches'/CASE;patchdir.mkdir(parents=True,exist_ok=True)
    tables=bytes(tags.get(347,b'')); table=tables[2:-2] if tables.endswith(b'\xff\xd9') else tables[2:]
    across=math.ceil(tags[256]/512)
    def fetch_tile(item):
        col,row,fraction=item; index=row*across+col
        offset,length=int(tags[324][index]),int(tags[325][index])
        target=patchdir/f'{col*2048}_{row*2048}.png'
        if not target.exists():
            data=fetch_range(URL,offset,offset+length-1)
            if table: data=data[:2]+table+data[2:]
            image=Image.open(io.BytesIO(data)).convert('RGB');assert image.size==(512,512)
            image.save(target)
        return {'patch_name':target.name,'x0':col*2048,'y0':row*2048,'source_level':2,
                'downsample':4,'tile_width':512,'footprint_width_level0':2048,'magnification':10,
                'tissue_fraction':fraction,'source_offset':offset,'source_bytes':length}
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(fetch_tile,selected))
    manifest={'case_id':'C1','loader_id':CASE,'slide_id':'test_001','source_url':URL,
              'level0_dimensions':[86016,89600],'nominal_native_magnification':40,'native_mpp':0.227273,
              'read_mpp':0.909092,'magnification_convention':'nominal 40x / downsample; objective field not present in XML',
              'downsample':4,'magnification':10,'candidate_count':len(records),'eligible_count':len(eligible),
              'selection':'Saturation/intensity tissue mask; >=25% tissue; deterministic uniform indices in row-major tile list. No expert coordinates or tumor labels used.',
              'wsi_description':desc,'patches':records}
    (BASE/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    geometry={CASE:{'coordinate_scale':4,'base_magnification':10,'coordinate_space':'CAMELYON16_level0',
                   'level0_verified':True,'case_id':'C1','slide_id':'test_001','slide_dimensions_level0':[86016,89600],
                   'nominal_native_magnification':40,'magnification_convention':'nominal 40x/downsample; XML pixel spacing ~0.227um at level0',
                   'downsample':4,'read_mpp':0.909092,'evidence':'Official TIFF IFD dimensions and 512px tile offsets; origins=tile indices*512*4; ImageMapping/ImageMetrics verified.'}}
    (BASE/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
    (BASE/'descriptions.json').write_text(json.dumps({CASE:{r['patch_name']:'' for r in records}},indent=2))
    (BASE/'questions.json').write_text(json.dumps([{'Id':CASE,'Question':'Does this lymph node slide contain metastatic carcinoma? Choose Malignant if metastatic carcinoma is present, otherwise Benign.', 'Choice':['Benign','Malignant'],'Answer':'Malignant'}],indent=2))
    # Candidate encoding is separate from the inference so PLIP is never on GPU.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,r'D:\PathCode\external\PLIP')
    from plip import PLIP
    plip=PLIP(r'D:\PathCode\checkpoints\plip');plip.model=plip.model.to('cpu');plip.device='cpu'
    features=BASE/'features'/CASE;features.mkdir(parents=True,exist_ok=True)
    images=[Image.open(patchdir/r['patch_name']).convert('RGB') for r in records]
    embeddings=plip.encode_images(images,batch_size=8)
    for record,emb in zip(records,embeddings):np.save(features/(Path(record['patch_name']).stem+'.npy'),emb)
    print('Prepared',len(records),'candidates, feature shape',embeddings.shape,flush=True)


if __name__=='__main__':main()
