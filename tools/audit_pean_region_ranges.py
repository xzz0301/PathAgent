"""Visual audit of existing grouping geometry; no rule or source-result changes."""
import argparse,base64,csv,hashlib,html,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Ellipse
from matplotlib.font_manager import FontProperties
from PIL import Image
from experts.pean_aligned import align_samples

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def picture(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main(root):
    out=root/'区域范围审计';out.mkdir(exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    cards=[];all_stats=[];provenance=[]
    for summary in load(root/'summary.json'):
        name=summary['slide_id'];p=root/name;source=p/'spatial/result.json';before=digest(source)
        current=load(source);meta=load(p/'recording_metadata.json');w,h=current['slide_dimensions']
        rows=[]
        for row in csv.DictReader((p/'samples.csv').open(encoding='utf-8-sig')):
            for k,v in list(row.items()):
                if v in ('True','False'):row[k]=v=='True'
                elif not v:row[k]=None
                else:
                    try:row[k]=float(v)
                    except ValueError:pass
            row['frame_index']=int(row['frame_index']);rows.append(row)
        uncompressed=align_samples(rows,(w,h),meta['fps_header'],do_compress=False)
        assert len(uncompressed['steps'])==current['before_compression']['visits']
        events=uncompressed['events'];valid=[e for e in events if e['input_valid']]
        assert sum(e['observation_mode']=='detailed_roi' for e in valid)==len(valid)
        displayed={r['source_region_id']:r['region_id'] for r in current['regions']}
        overview=Image.open(p/'overview.png');case_dir=out/name;case_dir.mkdir(exist_ok=True)
        figures=[];stats=[]
        for region in uncompressed['regions']:
            old_id=region['source_region_id'];box=region['detail_reference_box'];assert box is not None
            l,t,r,b=box;bw,bh=r-l,b-t
            coverage=max(0,min(w,r)-max(0,l))*max(0,min(h,b)-max(0,t))/(w*h)
            ids=set(region['source_event_ids']);points=[e for e in events if e['event_id'] in ids]
            anchor=(region.get('grouping_anchor_x_px',region['x_px']),region.get('grouping_anchor_y_px',region['y_px']))
            inside_ellipse=sum(((e['x_px']-anchor[0])/1500)**2+((e['y_px']-anchor[1])/2600)**2<1 for e in points)
            stat=dict(case=name,source_region_id=old_id,display_region_id=displayed.get(old_id),reference_box=box,
                      box_width=bw,box_height=bh,wsi_intersection_fraction=coverage,valid_frames=len(points),
                      outside_comparison_ellipse=len(points)-inside_ellipse,anchor_level0=anchor)
            stats.append(stat)
            fig,(ax,info)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained')
            ax.imshow(overview,extent=(0,w,h,0),zorder=0)
            ax.scatter([e['x_px'] for e in valid],[e['y_px'] for e in valid],s=6,c='#64748b',alpha=.22,zorder=2)
            ax.add_patch(Rectangle((0,0),w,h,fill=False,edgecolor='#334155',linewidth=1.1,zorder=3))
            ax.add_patch(Rectangle((l,t),bw,bh,fill=False,edgecolor='#b36913',linewidth=2.2,zorder=4))
            ax.scatter([e['x_px'] for e in points],[e['y_px'] for e in points],s=13,c='#21658b',alpha=.55,zorder=5)
            ax.add_patch(Ellipse(anchor,3000,5200,fill=False,edgecolor='#21658b',linestyle='--',linewidth=1.7,zorder=6))
            ax.scatter(*anchor,marker='*',s=140,color='#111827',edgecolor='white',linewidth=.8,zorder=7)
            xlo=min(0,l,anchor[0]-1500);xhi=max(w,r,anchor[0]+1500);ylo=min(0,t,anchor[1]-2600);yhi=max(h,b,anchor[1]+2600)
            pad=max(xhi-xlo,yhi-ylo)*.035;ax.set_xlim(xlo-pad,xhi+pad);ax.set_ylim(yhi+pad,ylo-pad);ax.set_aspect('equal')
            ax.set_xlabel('WSI level-0 X（px）');ax.set_ylabel('WSI level-0 Y（px）')
            ax.set_title('全片与真实参考窗口（包含片外部分）',fontsize=13)
            label=f"最终 R{displayed[old_id]:02d}" if old_id in displayed else '最终展示被压缩'
            text=f'''原区域 {old_id} · {label}

实际归并：细览参考窗口
矩形尺寸：{bw:,.0f} × {bh:,.0f} px
与整片相交面积：{coverage:.1%}
（分母为WSI矩形面积，不是组织面积）

归入本区域：{len(points)} 个有效帧
其中 {len(points)-inside_ellipse} 帧位于对照椭圆外
这不是筛除结果，只用于比较两种尺度

橙色实线：实际参考矩形
蓝色虚线：半径1500 × 2600px对照椭圆
蓝色散点：本区域原始有效眼动
灰色散点：其余有效眼动
黑色星号：固定归并锚点
深灰边框：原始WSI范围

归并还结合窗口重叠/缩放规则。
矩形不是病灶标注，也不是细胞级ROI。
本图未改动算法、坐标或精简结果。'''
            info.axis('off');info.text(.01,.99,text,va='top',fontsize=12,linespacing=1.65,transform=info.transAxes)
            fig.suptitle(f'{name} · 当前归并范围审计',fontsize=17)
            target=case_dir/f'region_{old_id:02d}.png';fig.savefig(target,dpi=145);plt.close(fig);figures.append((stat,target))
        first=stats[0]
        cards.append(f'<section><h2>{name}</h2><p>{len(valid)}个有效帧全部被判为细览。归并产生{len(stats)}个区域、{len(uncompressed["steps"])}次访问；几何压缩后剩{len(current["regions"])}个区域、{len(current["steps"])}次访问。</p><p><b>首个参考窗口覆盖整片矩形的{first["wsi_intersection_fraction"]:.1%}</b>。下面逐个展开查看所有压缩前区域，避免只展示保留下来的区域。</p>')
        for i,(s,target) in enumerate(figures):
            cards.append(f'<details {"open" if i==0 else ""}><summary>原区域{s["source_region_id"]} · {s["valid_frames"]}有效帧 · 窗口覆盖{s["wsi_intersection_fraction"]:.1%} · '+(f'最终R{s["display_region_id"]:02d}' if s['display_region_id'] else '最终被压缩')+'</summary>'+picture(target)+'</details>')
        cards.append('</section>');all_stats.extend(stats)
        assert digest(source)==before;provenance.append({'source':str(source),'sha256':before,'unchanged':True})
    (out/'audit.json').write_text(json.dumps({'regions':all_stats,'source_checks':provenance},ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PEAN区域范围审计</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;background:#f2f5f8;color:#203d50;font:18px/1.8 system-ui}section{background:white;padding:24px;margin:24px 0}img{max-width:100%;height:auto}summary{cursor:pointer;padding:12px;background:#edf2f6}details{margin:18px 0}</style><h1>PEAN：当前规则究竟合并了多大范围</h1><p>这是当前算法的空间审计，不是新算法结果。每图保留完整WSI与片外参考窗口，显示实际归并边界、原始有效落点与锚点。虚线椭圆仅作尺度对照，三例当前实际均采用细览窗口规则。</p><p>1500×2600px是椭圆半径，完整宽高为3000×5200px。当前更大的问题是固定35000px短边门槛把接近全片的窗口也判为细览。面积比例仅用于审计，不是已经验证的新筛选阈值。</p>'+''.join(cards)
    (out/'区域范围审计.html').write_text(page,encoding='utf-8');print(json.dumps({'regions':len(all_stats),'output':str(out),'first_region_coverage':{s['case']:s['wsi_intersection_fraction'] for s in all_stats if s['source_region_id']==1}},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
