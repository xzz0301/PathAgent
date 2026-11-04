"""Prepare a fixed image-only candidate pool and blinded questions for a small pilot."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import openslide
from wsi_regions import WSIReader

CASES=[('C1','test_001'),('C2','test_008'),('C6','test_036'),('C8','test_045'),('C12','test_053'),('C48','test_029')]
QUESTION='Does this lymph node slide contain metastatic carcinoma? Choose Malignant if metastatic carcinoma is present, otherwise Benign.'
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--wsi-root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--c1-existing',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    labels={r['image']:r for r in csv.DictReader(a.reference.open(encoding='utf-8-sig'))}
    manifests={};geometries={};questions=[];evaluation=[];candidates={}
    for case,slide_id in CASES:
        key='DX1_'+case;path=a.wsi_root/(slide_id+'.tif');folder=a.output/'patches'/key;folder.mkdir(parents=True)
        with openslide.OpenSlide(str(path)) as slide:
            dims=list(slide.dimensions);overview=slide.get_thumbnail((2400,2400)).convert('RGB')
            manifest={'path':str(path.resolve()),'slide_id':slide_id,'dimensions':dims,'quickhash':slide.properties['openslide.quickhash-1'],
                      'nominal_native_magnification':40,'magnification_source':'CAMELYON16 nominal 40x convention; physical MPP from OpenSlide'}
        overview.save(a.output/(case+'_overview.png'))
        manifests[key]=manifest
        geometries[key]={'coordinate_scale':4,'base_magnification':10,'coordinate_space':'CAMELYON16_level0','level0_verified':True,
                         'case_id':case,'slide_id':slide_id,'slide_dimensions_level0':dims,'nominal_native_magnification':40,'wsi_quickhash':manifest['quickhash']}
        rgb=np.asarray(overview);hsv=np.asarray(overview.convert('HSV'));mask=(hsv[:,:,1]>35)&(rgb.min(2)<220)&(rgb.max(2)>50)
        eligible=[]
        for y in range(0,dims[1]-2048+1,2048):
            for x in range(0,dims[0]-2048+1,2048):
                l,t=int(x/dims[0]*mask.shape[1]),int(y/dims[1]*mask.shape[0])
                r,b=math.ceil((x+2048)/dims[0]*mask.shape[1]),math.ceil((y+2048)/dims[1]*mask.shape[0])
                coverage=float(mask[t:b,l:r].mean())
                if coverage>=.25:eligible.append((x,y,coverage))
        if not eligible:raise ValueError('No tissue candidates: '+case)
        selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(64,len(eligible)),dtype=int)]
        if case=='C1':
            selected=[(*map(int,f.stem.split('_')),None) for f in sorted(a.c1_existing.glob('*.png'))]
            if len(selected)!=64:raise ValueError('Expected existing 64 C1 candidates')
        records=[]
        with WSIReader(manifest) as reader:
            for x,y,coverage in selected:
                name=f'{x}_{y}.png'
                if case=='C1':
                    # Preserve exactly the already-reviewed C1 candidate pixels.
                    (folder/name).write_bytes((a.c1_existing/name).read_bytes());meta={'image_source':'existing_verified_c1_patch'}
                else:
                    image,meta=reader.read(dict(x0=x,y0=y,width0=2048,height0=2048),10);image.save(folder/name)
                records.append({'name':name,'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest(),'tissue_fraction':coverage,'acquisition':meta})
        candidates[key]={'eligible':len(eligible),'selected':len(records),'selection':'existing C1 pool; otherwise equally spaced row-major tissue-only candidates; no expert or tumour coordinates','patches':records}
        questions.append({'Id':key,'Question':QUESTION,'Choice':['Benign','Malignant'],'Answer':'WITHHELD'})
        label=labels[slide_id+'.tif']
        evaluation.append({'case_id':case,'long_id':key,'slide_id':slide_id,'label':'Malignant' if label['type']=='tumor' else 'Benign','official':label})
    for name,obj in [('wsi_manifest.json',manifests),('geometry.json',geometries),('questions_blinded.json',questions),('evaluation_labels.json',evaluation),('candidates.json',candidates)]:save(a.output/name,obj)
    save(a.output/'cohort.json',{'cases':CASES,'fixed_before_inference':True,'reference_url':'https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16/evaluation/reference.csv','reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'seed':128,'initial_sample_count':3,'max_attempts':5,'minimum_observation_rounds':1,'scope':'six-case development pilot; 64 image-only candidates per WSI, not full-slide diagnostic validation'})
    print('PREPARED_6_CASES',flush=True)

if __name__=='__main__':main()
