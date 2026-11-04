"""Paired fixed-ROI replay: same Patho-R1 weights/prompts, old pixels vs original WSI."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from wsi_regions import WSIReader,observation_roi

def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    baseline_hash=digest(a.baseline)
    baseline=json.loads(a.baseline.read_text());manifest=json.loads(a.manifest.read_text())[baseline['long_id']]
    originals=[];fresh=[];paired=[]
    with WSIReader(manifest) as reader:
        for row in baseline['trajectory']:
            path=Path(row['input_image_path']);assert digest(path)==row['input_image_sha256']
            old=Image.open(path).convert('RGB');new,acquisition=reader.read(observation_roi(row,baseline['geometry']),row['magnification'])
            oldpath=a.output/f"old_{row['step']:02d}.png";newpath=a.output/f"wsi_{row['step']:02d}.png"
            old.save(oldpath);new.save(newpath);originals.append(old);fresh.append(new)
            paired.append({'step':row['step'],'roi':acquisition['roi'],'old_size':list(old.size),'new_size':list(new.size),'acquisition':acquisition,'old_path':str(oldpath),'new_path':str(newpath)})
    save(a.output/'acquisition_checks.json',paired)
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from models.inference import patho_r1_describe
    from models.grounding import infer_evidence,map_regions
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(a.checkpoint),torch_dtype=torch.bfloat16,device_map='cuda:1');model.eval()
    processor=AutoProcessor.from_pretrained(str(a.checkpoint));newtrace=[]
    for row,pair,old,new in zip(baseline['trajectory'],paired,originals,fresh):
        for label,image in [('old',old),('wsi',new)]:
            torch.manual_seed(128);torch.cuda.manual_seed_all(128);audit={};started=time.perf_counter()
            text=patho_r1_describe(image,question=baseline['question'],patho_r1_processor=processor,patho_r1_model=model,coords=(row['x'],row['y']),magnification=row['magnification'],choices=baseline['choices'],max_new_tokens=1024,input_audit=audit)
            pair[label+'_description']=text;pair[label+'_model_input']=audit;pair[label+'_seconds']=time.perf_counter()-started
        torch.manual_seed(128);torch.cuda.manual_seed_all(128)
        proposal=infer_evidence(new,baseline['question'],model,processor,max_new_tokens=1024)
        pair['localization_proposal']=map_regions(proposal,pair['roi'])
        newpath=Path(pair['new_path'])
        newtrace.append({**row,'input_image_path':str(newpath),'input_image_sha256':digest(newpath),'width':new.width,'height':new.height,
                         'description':pair['wsi_description'],'model_input':pair['wsi_model_input'],'roi':pair['roi'],'acquisition':pair['acquisition']})
        save(a.output/'paired_results.json',paired)
        print(json.dumps({'step':row['step'],'old_size':old.size,'wsi_size':new.size,'schema_valid':proposal['schema_valid']},ensure_ascii=False),flush=True)
    result={**baseline,'trajectory':newtrace,'geometry':{**baseline['geometry'],'image_backend':'wsi','wsi_quickhash':manifest['quickhash']},
            'runtime':{**baseline['runtime'],'image_backend':'wsi','zoom_source':'original_wsi'},'pred_answer':None,'explanation':None,'ground_truth':'WITHHELD',
            'replay':{'mode':'fixed_roi_image_source_comparison','source_result':str(a.baseline),'source_sha256':baseline_hash,'final_classification_rerun':False}}
    result.pop('process',None);result.pop('trajectory_files',None)
    save(a.output/'replayed_roi.json',result)
    save(a.output/'complete.json',{'observations':len(paired),'baseline_unchanged':digest(a.baseline)==result['replay']['source_sha256'],'localization_schema_valid':sum(x['localization_proposal']['schema_valid'] for x in paired),'clinical_localization_accepted':False})
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    page=Image.new('RGB',(1160,len(paired)*610+60),'white');draw=ImageDraw.Draw(page)
    for i,(pair,old,new) in enumerate(zip(paired,originals,fresh)):
        y=60+i*610
        for x,image,title in [(25,old,'Old patch / crop'),(605,new,'Original WSI read')]:
            draw.text((x,y-40),f"{pair['step']:02d} {title}: {image.width} x {image.height}",fill='#203d50',font=font)
            page.paste(image.resize((512,512)),(x,y))
    page.save(a.output/'paired_inputs.png')
if __name__=='__main__':main()
