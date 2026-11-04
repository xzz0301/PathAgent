"""Offline localization of exact observed images; never feeds back to navigation."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from PIL import Image,ImageDraw
from experts.pean import dump,font
from models.grounding import infer_evidence,map_regions


def render_roi(image,evidence,path,step):
    scale=min(900/image.width,900/image.height)
    shown=image.resize((round(image.width*scale),round(image.height*scale)))
    canvas=Image.new("RGB",(1650,max(1000,shown.height+200)),"white")
    canvas.paste(shown,(30,110));draw=ImageDraw.Draw(canvas)
    draw.text((30,20),f"Observation {step:02d} | model-proposed evidence",font=font(28),fill="#16354c")
    draw.text((950,110),"Schema: "+("valid" if evidence["schema_valid"] else "failed"),font=font(24),fill="#16354c")
    yy=165
    import textwrap
    for region in evidence["regions"]:
        l,t,r,b=region["bbox_norm"]
        colour="#246590"
        draw.rectangle((30+l*shown.width,110+t*shown.height,30+r*shown.width,110+b*shown.height),outline=colour,width=4)
        draw.text((30+l*shown.width,110+t*shown.height),region["id"],font=font(22),fill=colour,stroke_width=2,stroke_fill="white")
        lines=textwrap.wrap(region["id"]+": "+region["finding"],48)
        for line in lines:
            draw.text((950,yy),line,font=font(20),fill="#243b4c");yy+=29
        fraction=(r-l)*(b-t)
        draw.text((950,yy),f"Area: {100*fraction:.1f}% of ROI"+(" | broad box" if fraction>.5 else ""),font=font(19),fill="#805b17")
        yy+=29
        yy+=15
    if not evidence["regions"]:
        message=evidence.get("parse_error") or evidence.get("no_evidence_reason") or "No evidence localized"
        for line in textwrap.wrap(message,48):
            draw.text((950,yy),line,font=font(20),fill="#805b17");yy+=29
    draw.text((30,canvas.height-45),"Model proposals; morphology and diagnostic relevance require review.",font=font(22),fill="#805b17")
    canvas.save(path,dpi=(300,300))


def render_wsi(wsi,observations,path,mode="focus"):
    import openslide
    with openslide.OpenSlide(str(wsi)) as slide:
        dimensions=slide.dimensions
        overview=slide.get_thumbnail((1500,1400)).convert("RGB")
    canvas=Image.new("RGB",(2300,max(1550,220+len(observations)*100)),"white")
    draw=ImageDraw.Draw(canvas)
    title="PathAgent | ROI navigation" if mode=="roi" else "PathAgent | evidence focus"
    draw.text((35,25),title,font=font(32),fill="#16354c")
    subtitle=("Blue: ROI visit order, represented by ROI centres; grey: actual image footprint."
              if mode=="roi" else "Blue: evidence focus route; grey: observed ROI footprint. No within-ROI temporal ordering.")
    draw.text((35,80),subtitle,font=font(22),fill="#526572")
    scale=min(1450/dimensions[0],1330/dimensions[1])
    w,h=round(dimensions[0]*scale),round(dimensions[1]*scale)
    offset=(35+(1450-w)//2,150+(1330-h)//2)
    canvas.paste(overview.resize((w,h)),offset)
    positions=[]
    for row in observations:
        roi=row["roi"]
        draw.rectangle((offset[0]+roi["x0"]*scale,offset[1]+roi["y0"]*scale,
                        offset[0]+(roi["x0"]+roi["width0"])*scale,
                        offset[1]+(roi["y0"]+roi["height0"])*scale),outline="#8397a2",width=2)
        primary=None
        if mode=="roi":
            primary=(offset[0]+(roi["x0"]+roi["width0"]/2)*scale,
                     offset[1]+(roi["y0"]+roi["height0"]/2)*scale,row["step"])
        for region in row["evidence"]["regions"]:
            if mode=="roi":continue
            l,t,r,b=region["bbox_level0"]
            draw.rectangle((offset[0]+l*scale,offset[1]+t*scale,offset[0]+r*scale,offset[1]+b*scale),outline="#246590",width=2)
            if region["is_primary"]:
                x,y=region["point_level0"]
                primary=(offset[0]+x*scale,offset[1]+y*scale,row["step"])
        positions.append(primary)  # Keep missing observations as explicit gaps.
    import math
    for a,b in zip(positions,positions[1:]):
        if a is None or b is None:continue
        length=math.dist(a[:2],b[:2])
        if length<3:continue
        ux,uy=(b[0]-a[0])/length,(b[1]-a[1])/length
        end=(b[0]-ux*12,b[1]-uy*12)
        draw.line((a[:2],end),fill="#246590",width=4)
        draw.polygon([end,(end[0]-ux*13+uy*6,end[1]-uy*13-ux*6),
                      (end[0]-ux*13-uy*6,end[1]-uy*13+ux*6)],fill="#246590")
    label_boxes=[]
    for pos in positions:
        if pos:
            x,y,step=pos
            draw.ellipse((x-13,y-13,x+13,y+13),fill="#246590",outline="white",width=2)
            # Move labels, never data points. Leader lines preserve nearby zoom visits.
            for dx,dy in ((18,-18),(18,22),(-42,-18),(-42,22),(35,-48),(35,48),(-62,-48),(-62,48)):
                box=draw.textbbox((x+dx,y+dy),str(step),font=font(21),stroke_width=2)
                padded=(box[0]-5,box[1]-5,box[2]+5,box[3]+5)
                if not any(padded[0]<b[2] and padded[2]>b[0] and padded[1]<b[3] and padded[3]>b[1] for b in label_boxes):break
            label_boxes.append(padded)
            draw.line((x,y,(box[0]+box[2])/2,(box[1]+box[3])/2),fill="#526572",width=1)
            draw.text((x+dx,y+dy),str(step),font=font(21),fill="#16354c",stroke_width=2,stroke_fill="white")
    heading="Observation / nominal magnification / visit" if mode=="roi" else "Observation / nominal magnification / evidence"
    draw.text((1530,150),heading,font=font(24),fill="#16354c")
    for i,row in enumerate(observations):
        y=210+i*100
        count="ROI visit" if mode=="roi" else f"{len(row['evidence']['regions'])} regions"
        draw.text((1530,y),f"{row['step']:02d}    {row.get('magnification','?')}x    {count}",font=font(23),fill="#16354c")
        status=("Actual observation footprint recorded" if mode=="roi" else
                "parsed; morphology unreviewed" if row["evidence"]["schema_valid"] else "localization failed")
        draw.text((1530,y+37),status,font=font(20),fill="#805b17")
    footer=("ROI centres indicate visit order. Cropping magnification does not add image resolution." if mode=="roi" else
            "Focus points are model-selected evidence locations, not physiological gaze measurements.")
    draw.text((35,canvas.height-40),footer,font=font(22),fill="#526572")
    canvas.save(path,dpi=(300,300))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wsi",type=Path)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--max-observations",type=int,default=0)
    parser.add_argument("--max-new-tokens",type=int,default=1024)
    parser.add_argument("--mirror-probe",action="store_true")
    parser.add_argument("--coordinate-space",choices=("normalized","input_pixels"),default="normalized")
    args=parser.parse_args()
    result=json.loads(args.result.read_text(encoding="utf-8"))
    original_result_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest()
    if not result.get("geometry",{}).get("level0_verified"):
        raise ValueError("Verified level-0 geometry is required")
    trajectory=result["trajectory"][:args.max_observations or None]
    if not trajectory or any(not r.get("input_image_path") for r in trajectory):
        raise ValueError("Exact saved observation images are required; rerun with capture-observations")
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.checkpoint),torch_dtype=torch.bfloat16,device_map=args.device)
    model.eval()
    processor=AutoProcessor.from_pretrained(str(args.checkpoint))
    observations=[]
    started=time.perf_counter()
    for row in trajectory:
        path=Path(row["input_image_path"])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row["input_image_sha256"]:raise ValueError("Observation snapshot hash changed")
        image=Image.open(path).convert("RGB")
        grounding_image=image
        grounding_snapshot=None
        if args.coordinate_space=="input_pixels":
            # Qwen vision uses 28-pixel multiples. Declare and save this exact canvas.
            size=tuple(max(28,round(v/28)*28) for v in image.size)
            grounding_image=image.resize(size,Image.Resampling.LANCZOS)
            grounding_snapshot=args.output/f"grounding_input_{row['step']:04d}.png"
            grounding_image.save(grounding_snapshot)
        from wsi_regions import observation_roi
        roi=observation_roi(row,result['geometry'])
        before=time.perf_counter()
        evidence=infer_evidence(grounding_image,result["question"],model,processor,max_new_tokens=args.max_new_tokens,coordinate_space=args.coordinate_space)
        evidence=map_regions(evidence,roi)
        evidence["inference_seconds"]=time.perf_counter()-before
        item={"step":row["step"],"magnification":row.get("magnification"),"roi":roi,
              "input_image_sha256":digest,"input_image_path":str(path),"evidence":evidence}
        if grounding_snapshot:
            item["grounding_canvas"]={"path":str(grounding_snapshot),"sha256":hashlib.sha256(grounding_snapshot.read_bytes()).hexdigest(),
                                     "original_size":list(image.size),"canvas_size":list(grounding_image.size),
                                     "operation":"uniform resize to declared model canvas; normalized boxes map back to original ROI"}
        if args.mirror_probe and row==trajectory[0]:
            from PIL import ImageOps
            probe=infer_evidence(ImageOps.mirror(grounding_image),result["question"],model,processor,max_new_tokens=args.max_new_tokens,coordinate_space=args.coordinate_space)
            item["horizontal_mirror_probe"]=probe
        observations.append(item)
        dump(args.output/"progress.json",{"completed":len(observations),"total":len(trajectory),"observations":observations})
        render_roi(image,evidence,args.output/f"observation_{row['step']:04d}.png",row["step"])
        print(json.dumps({"step":row["step"],"schema_valid":evidence["schema_valid"],"regions":len(evidence["regions"])},ensure_ascii=False),flush=True)
    receipt={"mode":"record_only_offline","navigation_feedback":False,
             "source_result":str(args.result.resolve()),"source_result_sha256":original_result_sha256,
             "checkpoint":str(args.checkpoint.resolve()),"model_class":type(model).__name__,
             "dtype":str(model.dtype),"device":str(model.device),"torch_version":torch.__version__,
             "observations":observations,"elapsed_seconds":time.perf_counter()-started,
             "schema_valid_count":sum(r["evidence"]["schema_valid"] for r in observations),
             "localized_observations":sum(bool(r["evidence"]["regions"]) for r in observations),
             "clinical_validation":"not_performed",
             "source_navigation_unchanged":hashlib.sha256(args.result.read_bytes()).hexdigest()==original_result_sha256}
    dump(args.output/"evidence.json",receipt)
    if args.wsi:
        render_wsi(args.wsi,observations,args.output/"wsi_evidence_trajectory.png")
        render_wsi(args.wsi,observations,args.output/"wsi_roi_navigation.png",mode="roi")
    links="".join(f'<h2>Observation {r["step"]}</h2><img src="observation_{r["step"]:04d}.png">' for r in observations)
    full='<h2>实际ROI访问顺序（中心仅作显示）</h2><img src="wsi_roi_navigation.png"><h2>模型证据聚焦位置</h2><img src="wsi_evidence_trajectory.png">' if args.wsi else ''
    import html
    checkpoint_label=html.escape(args.checkpoint.name)
    (args.output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>PathAgent evidence localization</title><style>body{font:17px/1.6 sans-serif;margin:30px;color:#16354c;background:#f4f7fa}img{max-width:100%}</style><h1>PathAgent 证据聚焦定位</h1><p>定位模型：'+checkpoint_label+'；离线定位，不反馈导航。以下框为模型提出的证据，尚未经病理语义复核。失败结果不补中心点。</p>'+full+links,encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!="observations"},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
