"""Bounded magnification-aware PEAN experiment, preserving the established output."""
import argparse,base64,csv,json,sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from experts.observations import Config
from experts.pean_aligned import align_samples,render
from experts.pean import font

@dataclass(frozen=True)
class DynamicConfig(Config):
    dynamic_ellipse:bool=True
    ellipse_width_px:float=150.
    ellipse_height_px:float=150.

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main(source,out):
    out.mkdir(exist_ok=True);cards=[];summary=[]
    for case in load(source/'summary.json'):
        name=case['slide_id'];p=source/name;m=load(p/'recording_metadata.json');w,h=m['slide_dimensions'];ratio=w/h
        ew,eh=(150*ratio,150) if ratio>=1 else (150,150/ratio)
        cfg=DynamicConfig(ellipse_width_px=ew,ellipse_height_px=eh)
        rows=[]
        for row in csv.DictReader((p/'samples.csv').open(encoding='utf-8-sig')):
            for k,v in list(row.items()):
                if v in ('True','False'):row[k]=v=='True'
                elif not v:row[k]=None
                else:
                    try:row[k]=float(v)
                    except ValueError:pass
            row['frame_index']=int(row['frame_index']);rows.append(row)
        overview=Image.open(p/'overview.png').convert('RGB');dest=out/name;dest.mkdir(exist_ok=True)
        cards.append(f'<section><h2>{name}</h2><p>WSI {w}×{h}；宽高比{ratio:.3f}。20倍完整轴长{ew:.1f}×{eh:.1f}px（不是半径）。</p>')
        for mode in ['frame_order','fps_estimate']:
            r=align_samples(rows,(w,h),m['fps_header'],mode,do_compress=False,grouping_config=cfg)
            events={e['event_id']:e for e in r['events']}
            for reg in r['regions']:
                anchor=events[reg['grouping_anchor_event_id']];z0=anchor['viewer_relative_zoom'];ranges=[]
                for eid in reg['source_event_ids']:
                    e=events[eid];z=max(z0,e['viewer_relative_zoom']);rx,ry=ew/(2*z),eh/(2*z)
                    assert ((e['x_px']-anchor['x_px'])/rx)**2+((e['y_px']-anchor['y_px'])/ry)**2<1+1e-9
                    e['effective_ellipse_radii_level0']=[rx,ry]
                    if [rx,ry] not in ranges:ranges.append([rx,ry])
                reg['ellipse_radii_level0']=ranges
            r['method_note']='Displayed-pixel extent fixed at the 20x reference dimensions; level0 radius = axis_pixels * actual downsample / 2. Compare with the smaller of anchor/current footprints. No whole-viewport grouping; no legacy 1500x2600 geometric compression.'
            (dest/(mode+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
            # Retain the familiar complete-WSI route and right-hand visit table.
            route=dest/(mode+'_route.png');render(overview,(w,h),r,route,name,40.)
            canvas=Image.new('RGB',(1800,1600),'white');draw=ImageDraw.Draw(canvas);scale=min(1680/w,1360/h);size=(round(w*scale),round(h*scale));ox,oy=(1800-size[0])//2,150
            canvas.paste(overview.resize(size,Image.Resampling.LANCZOS),(ox,oy))
            positions={reg['region_id']:(ox+reg['x_px']*scale,oy+reg['y_px']*scale) for reg in r['regions']}
            import math
            for edge in r['edges']:
                a,b=positions[edge['from_region']],positions[edge['to_region']];distance=math.dist(a,b)
                if distance<5:continue
                ux,uy=(b[0]-a[0])/distance,(b[1]-a[1])/distance
                draw.line((a,b),fill='white',width=4);draw.line((a,b),fill='#246590',width=2)
                draw.polygon([b,(b[0]-ux*12+uy*5,b[1]-uy*12-ux*5),(b[0]-ux*12-uy*5,b[1]-uy*12+ux*5)],fill='#246590')
            draw.text((35,25),name+' | Dynamic ellipse boundaries',font=font(30),fill='#203d50')
            draw.text((35,75),f'20x full axes: {ew:.1f} x {eh:.1f} px | all retained region boundaries',font=font(24),fill='#203d50')
            for reg in r['regions']:
                x=ox+reg['grouping_anchor_x_px']*scale;y=oy+reg['grouping_anchor_y_px']*scale
                for rx,ry in reg['ellipse_radii_level0']:
                    draw.ellipse((x-rx*scale,y-ry*scale,x+rx*scale,y+ry*scale),outline='#246590',width=2)
                for eid in reg['source_event_ids']:
                    e=events[eid];px,py=ox+e['x_px']*scale,oy+e['y_px']*scale;draw.ellipse((px-2,py-2,px+2,py+2),fill='#246590')
                px,py=positions[reg['region_id']];draw.ellipse((px-4,py-4,px+4,py+4),fill='#246590',outline='white')
                draw.text((px+6,py-20),f"R{reg['region_id']:02d}",font=font(17),fill='#203d50',stroke_width=2,stroke_fill='white')
            draw.text((35,1540),'Concentric ellipses show magnification-dependent bounds; no diagnostic annotation.',font=font(24),fill='#526572')
            boundary=dest/(mode+'_boundaries.png');canvas.save(boundary)
            title='空间版：时间未校准' if mode=='frame_order' else '50FPS估计150ms版：非实测时长'
            cards.append(f'<h3>{title}</h3><p>{len(r["regions"])}区域 / {len(r["steps"])}访问。此次不叠加旧版大范围折线压缩，避免再次吞并局部观察。</p>')
            for pic in [route,boundary]:cards.append('<img src="data:image/png;base64,'+base64.b64encode(pic.read_bytes()).decode()+'">')
            summary.append(dict(case=name,mode=mode,width20=ew,height20=eh,regions=len(r['regions']),visits=len(r['steps'])))
        cards.append('</section>')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'PEAN动态椭圆对照.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>body{max-width:1500px;margin:30px auto;padding:24px;background:#f2f5f8;color:#203d50;font:18px/1.8 system-ui}section{background:white;padding:24px;margin:24px 0}img{max-width:100%}</style><h1>PEAN：动态椭圆归并对照</h1><p>20倍短轴150px，长轴按每张WSI宽高比确定。采用当前与首次锚点两种倍率下较小的范围判断，避免低倍大范围在放大后吞并新关注位置。倍率变化仍记录在访问序列；矩形视口不再替代眼动区域。图中的同心椭圆对应实际使用过的不同倍率范围。原始质量筛选不变；估计时间仍非实测。本页是待讨论的算法对照，不代表临床验证。</p>'+''.join(cards)+'</html>',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
