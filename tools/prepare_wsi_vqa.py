"""Prepare the complete server WSI-VQA test set at calibrated nominal 5x.

No fixed candidate cap. Tissue segmentation is an explicitly recorded Otsu
saturation grid implementation; exact unpublished Trident settings are unknown.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def positions(length, span):
    if length < span:
        raise ValueError('Slide smaller than physical patch field')
    return sorted(set(range(0, length-span+1, span)) | {length-span})


def main():
    import cv2
    import numpy as np
    import openslide
    from wsi_regions import WSIReader, file_fingerprint

    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--uncalibrated-case', action='append', default=[],
                   help='Explicitly authorized case ID to retain as missing when objective metadata is absent')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    original_path = a.dataset/'metadata/WsiVQA_test.json'
    mapped_path = a.dataset/'metadata/questions_with_slide_paths.json'
    questions = json.loads(original_path.read_text())
    mapped = json.loads(mapped_path.read_text())
    assert len(questions) == len(mapped) == 735
    assert all({k:v for k,v in m.items() if k != 'slide_path'} == q for m,q in zip(mapped,questions))
    assert len({(q['Id'], q['Question']) for q in questions}) == 735
    case_paths = {}
    for q in mapped:
        previous = case_paths.setdefault(q['Id'], q['slide_path'])
        assert previous == q['slide_path'] and Path(previous).is_file()
    assert len(case_paths) == 86
    save(a.output/'reference.json', questions)
    protocol = {'test_questions':735,'slides':86,'closed':390,'open':345,
                'question_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
                'mapped_sha256':hashlib.sha256(mapped_path.read_bytes()).hexdigest(),
                'base_magnification':5,'patch_pixels_at_5x':512,'candidate_cap':None,
                'tissue_selection':'median saturation Otsu, 5x5 closing, full grid coverage >=20%; edge-covering origins',
                'preprocessing_difference':'Exact Trident experiment configuration not published; explicitly recorded reconstruction.',
                'example_cases_fixed_before_inference':['TCGA-BH-A202','TCGA-D8-A27M','TCGA-LD-A74U'],
                'inference_labels':'WITHHELD; reference separate and used only by evaluator'}
    save(a.output/'protocol.json', protocol)
    authorized_missing=set(a.uncalibrated_case)
    if not authorized_missing.issubset(case_paths):
        raise ValueError('Unknown uncalibrated case ID')
    manifests, geometries, candidates = {}, {}, {}
    missing=[]
    previous={name:json.loads((a.output/name).read_text()) if (a.output/name).exists() else {}
              for name in ['wsi_manifest.json','geometry.json','candidates.json']}
    for index, (case,path) in enumerate(sorted(case_paths.items())):
        key = Path(path).stem
        # Resume completed patches without retiling or rewriting images. Verify
        # immutable image hashes before accepting the previous per-slide receipt.
        prior=previous['candidates.json'].get(key)
        if not a.inventory_only and prior and all('sha256' in row for row in prior['patches']):
            manifest=previous['wsi_manifest.json'][key]
            if manifest['path']!=path or manifest['nominal_native_magnification'] not in [20,40]:
                raise ValueError('Previous slide identity/calibration differs: '+case)
            with WSIReader(manifest):
                pass
            folder=a.output/'patches'/key
            if {p.name for p in folder.glob('*.png')}!={row['name'] for row in prior['patches']}:
                raise ValueError('Previous candidate file set differs: '+case)
            for row in prior['patches']:
                if hashlib.sha256((folder/row['name']).read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('Previous candidate content changed: '+case+'/'+row['name'])
            manifests[key]=manifest
            geometries[key]=previous['geometry.json'][key]
            candidates[key]=prior
            print(json.dumps({'case':case,'reused_verified_candidates':len(prior['patches'])}),flush=True)
            continue
        with openslide.OpenSlide(path) as slide:
            dims = list(slide.dimensions)
            raw = slide.properties.get('openslide.objective-power')
            if raw is None:
                if case in authorized_missing:
                    missing.append({'case_id':case,'slide_path':path,'reason':'missing_objective_magnification',
                                    'dimensions':dims,'metadata_checked':['openslide.objective-power','aperio.AppMag','MPP'],
                                    'policy':'Do not infer magnification; all questions retained as missing in full-test statistics'})
                    continue
                raise ValueError('No verified objective magnification: '+case)
            native = float(raw)
            if native not in [20,40]:
                raise ValueError('Uncalibrated scanner objective: '+str(native))
            manifest = {'path':path,'slide_id':case,'dimensions':dims,
                        'quickhash':slide.properties.get('openslide.quickhash-1'),
                        'nominal_native_magnification':native,'magnification_source':'openslide.objective-power'}
            overview = slide.get_thumbnail((3000,3000)).convert('RGB')
        if not manifest['quickhash']:
            manifest['file_fingerprint']=file_fingerprint(path)
            manifest['identity_mode']='file size + head/tail 1MiB SHA256; not full-file SHA256'
        overview_dir = a.output/'overviews'; overview_dir.mkdir(exist_ok=True)
        overview.save(overview_dir/(case+'.png'))
        saturation = cv2.medianBlur(np.asarray(overview.convert('HSV'))[:,:,1], 7)
        _,mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)) > 0
        span = int(512*native/5)
        selected = []
        for y in positions(dims[1],span):
            for x in positions(dims[0],span):
                l,t = int(x/dims[0]*mask.shape[1]),int(y/dims[1]*mask.shape[0])
                r,b = math.ceil((x+span)/dims[0]*mask.shape[1]),math.ceil((y+span)/dims[1]*mask.shape[0])
                coverage = float(mask[t:b,l:r].mean())
                if coverage >= .2:
                    selected.append({'name':f'{x}_{y}.png','roi':dict(x0=x,y0=y,width0=span,height0=span),
                                     'tissue_fraction':coverage})
        if not selected:
            raise ValueError('No tissue candidates: '+case)
        manifests[key] = manifest
        geometries[key] = {'coordinate_scale':native/5,'base_magnification':5,
                           'coordinate_space':'WSI_level0','level0_verified':True,'case_id':case,
                           'slide_dimensions_level0':dims,'nominal_native_magnification':native}
        candidates[key] = {'case_id':case,'count':len(selected),'patches':selected}
        if not a.inventory_only:
            folder = a.output/'patches'/key; folder.mkdir(parents=True,exist_ok=True)
            with WSIReader(manifest) as reader:
                for row in selected:
                    target = folder/row['name']
                    if not target.exists():
                        image,_ = reader.read(row['roi'],5)
                        temporary=target.with_suffix('.writing')
                        image.save(temporary,format='PNG')
                        temporary.replace(target)
                    from PIL import Image
                    with Image.open(target) as check:
                        if check.size != (512,512):
                            raise ValueError('Existing candidate has wrong dimensions: '+str(target))
                        check.verify()
                    row['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        # Keep prior complete records present while a concurrent caption worker
        # reads already prepared slides; publish a complete manifest below.
        for n,obj in [('wsi_manifest.json',manifests),('geometry.json',geometries),('candidates.json',candidates)]:
            save(a.output/n,{**previous[n],**obj})
        save(a.output/'preparation_status.json', {'stage':'inventory' if a.inventory_only else 'patches',
             'slides_completed':index+1,'slides_total':86,'candidates_total':sum(c['count'] for c in candidates.values())})
        print(json.dumps({'case':case,'slides_completed':index+1,'candidate_count':len(selected)}),flush=True)
    for n,obj in [('wsi_manifest.json',manifests),('geometry.json',geometries),('candidates.json',candidates)]:
        save(a.output/n,obj)
    missing_ids={row['case_id'] for row in missing}
    eligible=[q for q in questions if q['Id'] not in missing_ids]
    save(a.output/'excluded_slides.json',missing)
    save(a.output/'questions_blinded.json',[{**q,'Answer':'WITHHELD'} for q in eligible])
    protocol.update(authorized_uncalibrated_cases=sorted(authorized_missing),
        excluded_slides=missing,eligible_slides=len(manifests),eligible_questions=len(eligible),
        eligible_closed=sum(bool(q.get('Choice')) for q in eligible),
        eligible_open=sum(not q.get('Choice') for q in eligible),
        missing_policy='Full 86-slide / 735-question reference frozen; separate calibrated-subset metrics')
    save(a.output/'protocol.json',protocol)
    save(a.output/'preparation_status.json', {'stage':'inventory_complete' if a.inventory_only else 'complete',
        'slides_completed':len(manifests),'slides_total':86,'excluded_slides':len(missing),
        'eligible_questions':len(eligible),'candidates_total':sum(c['count'] for c in candidates.values())})


if __name__ == '__main__':
    main()
