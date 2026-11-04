"""Render observed ROI footprints from an existing PathAgent run; no focus points or model inference."""
import argparse
import csv
import hashlib
import html
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsi_regions import observation_roi


INK='#203d50'
ROUTE='#d3463e'


def font(size):
    for name in ('C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def footprints(result,snapshot_root=None):
    geometry=result['geometry']
    if not geometry.get('level0_verified'):raise ValueError('Verified level-0 geometry required')
    scale=float(geometry['coordinate_scale']);width0,height0=geometry['slide_dimensions_level0']
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid coordinate scale')
    rows=[]
    for sequence,source in enumerate(result['trajectory'],1):
        if source['step']!=sequence or source['status']!='completed':
            raise ValueError('Complete, consecutive original observations required; no silent filtering')
        image_path=Path(source['input_image_path'])
        if snapshot_root:image_path=Path(snapshot_root)/image_path.name
        if digest(image_path)!=source['input_image_sha256']:raise ValueError('Input image hash mismatch')
        with Image.open(image_path) as im:w,h=im.size
        if [w,h]!=[source['width'],source['height']]:raise ValueError('Saved image dimensions disagree')
        roi=observation_roi(source,geometry)
        x,y,rw,rh=(roi[k] for k in ('x0','y0','width0','height0'))
        if not (0<=x<x+rw<=width0 and 0<=y<y+rh<=height0):raise ValueError('ROI outside slide')
        rows.append({'step':sequence,'original_step':source['step'],'action':source['action'],
                     'x0':x,'y0':y,'width0':rw,'height0':rh,'magnification':source['magnification'],
                     'input_width':w,'input_height':h,'input_image_sha256':source['input_image_sha256'],
                     'image_source':source.get('acquisition',{}).get('image_source','crop_existing_patch' if source['action']=='zoom_observation' else 'saved_patch'),
                     'snapshot':str(image_path.resolve())})
    if not rows:raise ValueError('No observations')
    return rows


def centre(box):return ((box[0]+box[2])/2,(box[1]+box[3])/2)


def edge(box,toward):
    x,y=centre(box);dx,dy=toward[0]-x,toward[1]-y
    tx=(box[2]-box[0])/2/abs(dx) if dx else math.inf
    ty=(box[3]-box[1])/2/abs(dy) if dy else math.inf
    t=min(tx,ty)
    return x+dx*t,y+dy*t


def arrow(draw,start,end):
    length=math.dist(start,end)
    if length<12:return
    ux,uy=(end[0]-start[0])/length,(end[1]-start[1])/length
    draw.line((start,end),fill='white',width=9)
    draw.line((start,end),fill=ROUTE,width=5)
    draw.polygon([end,(end[0]-ux*20+uy*9,end[1]-uy*20-ux*9),
                  (end[0]-ux*20-uy*9,end[1]-uy*20+ux*9)],fill=ROUTE)


def overlaps(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]


def draw_route(overview,dimensions,rows):
    overview=overview.copy().convert('RGB');draw=ImageDraw.Draw(overview)
    sx,sy=overview.width/dimensions[0],overview.height/dimensions[1]
    boxes=[(r['x0']*sx,r['y0']*sy,(r['x0']+r['width0'])*sx,(r['y0']+r['height0'])*sy) for r in rows]
    for a,b in zip(boxes,boxes[1:]):
        if overlaps(a,b):continue  # Nested visits remain explicit in labels/table, not an invented point.
        arrow(draw,edge(a,centre(b)),edge(b,centre(a)))
    for box in boxes:
        draw.rectangle(box,outline='white',width=8)
        draw.rectangle(box,outline=ROUTE,width=4)
    occupied=[]
    for row,box in zip(rows,boxes):
        label=f"{row['step']:02d}"
        candidates=[(box[2]+12,box[1]-8),(box[2]+12,box[3]+5),
                    (box[0]-60,box[1]-8),(box[0]-60,box[3]+5),(box[2]+60,box[1]-50)]
        for x,y in candidates:
            tag=(x-5,y-3,x+44,y+35)
            if not any(overlaps(tag,b) for b in occupied+boxes):break
        occupied.append(tag)
        anchor=edge(box,((tag[0]+tag[2])/2,(tag[1]+tag[3])/2))
        draw.line((anchor,((tag[0]+tag[2])/2,(tag[1]+tag[3])/2)),fill=ROUTE,width=2)
        draw.rectangle(tag,fill='white',outline=ROUTE,width=2)
        draw.text((x,y),label,font=font(26),fill=ROUTE)
    return overview


def render(overview,dimensions,rows,output,result):
    route=draw_route(overview,dimensions,rows)
    route.save(output/'roi_route.png',dpi=(300,300))
    page=Image.new('RGB',(route.width+880,max(route.height+210,700+len(rows)*190)),'white');page.paste(route,(35,140));d=ImageDraw.Draw(page)
    d.text((35,22),'PathAgent | Patho-R1 ROI navigation',font=font(35),fill=INK)
    d.text((35,76),'Actual observation footprints | arrows show visit order | no localized focus points',font=font(23),fill=INK)
    right=route.width+85
    case=result['geometry'].get('case_id',result.get('long_id',''))
    d.text((right,150),f"{case} / {result['geometry'].get('slide_id','WSI')}",font=font(29),fill=INK)
    d.text((right,205),'Observation     Nominal mag.     Image source',font=font(23),fill=INK)
    for i,row in enumerate(rows):
        y=260+i*190
        d.rectangle((right-10,y-8,page.width-30,y+160),fill='#f1f5f7')
        kind={'saved_patch':'Saved patch','original_wsi':'Direct WSI read','crop_existing_patch':'Existing patch crop'}[row['image_source']]
        d.text((right,y),f"{row['step']:02d}                 {row['magnification']:g}x         {kind}",font=font(24),fill=INK)
        d.text((right,y+45),f"Level-0 origin: ({row['x0']:g}, {row['y0']:g})",font=font(22),fill=INK)
        d.text((right,y+83),f"Footprint: {row['width0']:g} x {row['height0']:g} px",font=font(22),fill=INK)
        d.text((right,y+121),f"Input: {row['input_width']} x {row['input_height']} px",font=font(22),fill=INK)
    direct=all(r['image_source']=='original_wsi' for r in rows)
    notes=['Perceptor: Patho-R1-7B','Executor: Qwen3-4B | Navigator: PLIP',
           'Arrows and numbers show observation order.','Boxes preserve actual WSI footprints.',
           'Pixels read directly from original WSI.' if direct else 'Crop magnification is a requested setting;',
           'Per-observation geometry is recorded.' if direct else 'old-patch cropping adds no new WSI detail.']
    for i,line in enumerate(notes):d.text((right,310+len(rows)*190+i*43),line,font=font(23),fill=INK)
    d.text((35,page.height-45),'Source: original server run. No Qwen2.5-VL localization output is used.',font=font(23),fill=INK)
    page.save(output/'roi_route_with_table.png',dpi=(300,300))
    with (output/'roi_visits.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Patho-R1 ROI轨迹</title><style>body{max-width:1600px;margin:35px auto;padding:0 24px;font:18px/1.8 sans-serif;color:#203d50}img{max-width:100%}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ccc}</style><h1>Patho-R1 原始运行：ROI 访问轨迹</h1><p>框表示实际送入 Patho-R1 的图像在整张 WSI 中覆盖的范围；箭头表示访问顺序，不画精确观察点。编号沿用本次真实观察顺序。嵌套或重复访问保留各自编号；图像来源和实际范围见右表。</p><p>图像观察：Patho-R1-7B；执行：Qwen3-4B；导航：PLIP。当前图不使用 Qwen2.5-VL 的定位输出。</p><img src="roi_route_with_table.png" alt="左侧完整WSI上的ROI框与访问箭头，右侧列出各次观察的倍率、尺寸和来源"><p><a href="roi_route.png">仅全片框轨迹图</a> · <a href="roi_visits.csv">访问坐标与尺寸</a> · <a href="render_manifest.json">数据来源与核查</a></p></html>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--wsi',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot-root',type=Path)
    args=p.parse_args();source_hash=digest(args.result);result=json.loads(args.result.read_text(encoding='utf-8'))
    runtime=result.get('runtime',{});perceptor=runtime.get('weights',{}).get('perceptor','')
    if 'Patho-R1' not in perceptor:raise ValueError('This output requires a recorded Patho-R1 Perceptor run')
    rows=footprints(result,args.snapshot_root)
    import openslide
    with openslide.OpenSlide(str(args.wsi)) as slide:
        dimensions=list(slide.dimensions)
        if dimensions!=result['geometry']['slide_dimensions_level0']:raise ValueError('WSI dimensions mismatch')
        expected=result['geometry'].get('wsi_quickhash')
        if expected and slide.properties.get('openslide.quickhash-1')!=expected:raise ValueError('WSI identity mismatch')
        if not expected and args.wsi.stem!=result['geometry'].get('slide_id'):raise ValueError('WSI slide ID mismatch')
        overview=slide.get_thumbnail((2100,2100)).convert('RGB')
        quickhash=slide.properties.get('openslide.quickhash-1')
    args.output.mkdir(parents=True,exist_ok=False)
    overview.save(args.output/'wsi_overview.png')
    render(overview,dimensions,rows,args.output,result)
    manifest={'source_result':str(args.result.resolve()),'source_result_sha256':source_hash,
              'wsi':str(args.wsi.resolve()),'wsi_dimensions_level0':dimensions,'wsi_quickhash':quickhash,
              'models':runtime.get('weights'),'observations':rows,'qwen25_localization_used':False,
              'precise_focus_points_shown':False,'footprints_drawn_at_actual_scale':True,
              'source_result_unchanged':digest(args.result)==source_hash,
              'zoom_source':runtime.get('zoom_source'),'output_scope':'rerender of completed original run; no new inference',
              'script_sha256':digest(Path(__file__))}
    (args.output/'render_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'observations':len(rows),'output':str(args.output),'source_unchanged':manifest['source_result_unchanged']}))


if __name__=='__main__':main()
