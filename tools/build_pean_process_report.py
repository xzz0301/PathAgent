"""Self-contained PEAN report: every displayed decision traces to raw EPR fields."""
import argparse,base64,csv,html,json
from collections import Counter
from pathlib import Path

def esc(x):return html.escape(str(x if x is not None else '未提供'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def img(p):return '<img loading="lazy" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
REASONS={'background_at_overview_resolution':'概览组织掩膜外','outside_screen':'屏幕范围外','outside_wsi':'切片范围外','invalid_head_distance':'无效头部距离','invalid_pyramid_level':'无效金字塔层级','endpoint':'路径端点','geometric_turn':'保留空间转折或回返','long_valid_observation':'估计有效支持达到保护阈值','represented_by_bounded_segment':'在空间误差范围内由邻接线段表示','operation_evidence':'操作证据保护'}
def why(s):return '；'.join(REASONS.get(v,v) for v in str(s or '').split(';')) or '通过'
FIELDS=[('screen_x / screen_y','EPR逐帧整数','WSI图像在屏幕上的偏移；与eye_x/y作差后投影，不能当成眼动坐标'),('eye_x / eye_y','EPR逐帧整数','屏幕眼动落点；用于屏幕边界与坐标变换'),('epr_level','EPR逐帧整数','减去头部minlevel后索引NDPI实际金字塔downsample'),('head_z_mm','EPR逐帧浮点数','头部距离；非有限或≤0剔除，不当作注视置信度'),('timestamp_raw','EPR逐帧原始时间值','保留原值，时间单位尚未校准，不直接套用150ms'),('quickhash / slide_dimensions','EPR头部 + NDPI属性','匹配切片身份、尺寸和坐标范围'),('screen_dimensions / minlevel / fps_header','EPR头部','屏幕范围、层级偏移及50FPS估计假设'),('gaze_x/y_level0、valid_spatial_sample','程序派生字段','投影坐标与质量判断，不是EPR原生字段'),('source_event_ids / visit_id / region_id','算法派生字段','原始帧与归并访问、固定锚点之间的追溯关系')]

def raw_table(rows):
    keys=['frame_index','screen_x','screen_y','eye_x','eye_y','epr_level','head_z_mm','timestamp_raw','downsample','gaze_x_level0','gaze_y_level0','exclusion_reason']
    return table(['帧号（0起）','图像偏移X','图像偏移Y','眼动X','眼动Y','原层级','头距mm','原始时间','实际ds','全片X','全片Y','质量判断'],[[esc(r[k]) if k!='exclusion_reason' else esc(why(r[k])) for k in keys] for r in rows])

def build(root):
    summaries=load(root/'summary.json');parts=[]
    for summary in summaries:
        name=summary['slide_id'];p=root/name;m=load(p/'recording_metadata.json')
        rows=list(csv.DictReader((p/'samples.csv').open(encoding='utf-8-sig')));lookup={int(r['frame_index']):r for r in rows}
        spatial=load(p/'spatial/result.json');estimate=load(p/'fps_estimate_150ms/result.json')
        valid=[r for r in rows if r['valid_spatial_sample']=='True'];counts=Counter(r['exclusion_reason'] for r in rows if r['exclusion_reason'])
        anchor=spatial['regions'][0]['anchor_event_id'];r=lookup[anchor]
        parts.append(f'<section id="{name}"><h2>{esc(name)}</h2><p class="lead">{len(rows)}个原始帧 → {len(valid)}个空间有效帧 → {len(spatial["visits_audit"])}次归并访问 → {len(spatial["steps"])}次最终访问，涉及{len(spatial["regions"])}个区域。</p><h3>先看最终效果 · 时间未校准的空间版</h3>'+img(p/'spatial/trajectory.png'))
        audit_dir=root/'区域范围审计'/name
        if audit_dir.exists():
            parts.append('<h3>当前规则的范围审计：一个点实际代表多大区域</h3><p>橙色实线是实际细览参考窗口，蓝色散点是归入该区域的原始眼动；蓝色虚线是1500×2600px半径的对照椭圆，本例并未按该椭圆归并。矩形与整片的面积比例不是组织面积比例。以下涵盖全部压缩前区域，不修改现有算法或坐标。</p>')
            for i,figure in enumerate(sorted(audit_dir.glob('region_*.png'))):
                parts.append(f'<details {"open" if i==0 else ""}><summary>原区域{int(figure.stem.split("_")[-1])}：参考范围与全部原始落点</summary>'+img(figure)+'</details>')
        parts.append('<div class="step"><h3>Step 1 · 识别原始记录，将眼动落点放回切片</h3><p>输入是EPR二进制记录和对应NDPI，不是从最终轨迹倒推。以下表格中的原始列来自EPR解码；帧号是程序赋予的0起始序号。</p>')
        parts.append(table(['来源','本例实际数据'],[[esc('EPR文件'),esc(m['source_file'])],[esc('内部切片名'),esc(m['internal_slide_name'])],[esc('内部quickHash'),esc(m['quickhash'])],[esc('切片尺寸 / 屏幕尺寸'),esc(f"{m['slide_dimensions']} / {m['screen_dimensions']}")],[esc('minlevel / FPS头字段'),esc(f"{m['minlevel']} / {m['fps_header']}")]]))
        parts.append(f'<p><b>真实记录示例：帧{anchor}</b>，NDPI层级 = {r["epr_level"]} − {m["minlevel"]} = {int(r["epr_level"])-m["minlevel"]}；该层实际downsample = {r["downsample"]}。全片X = ({r["eye_x"]} − {r["screen_x"]}) × {r["downsample"]} = {r["gaze_x_level0"]}；全片Y = ({r["eye_y"]} − {r["screen_y"]}) × {r["downsample"]} = {r["gaze_y_level0"]}。</p>'+raw_table([r])+'</div>')
        parts.append('<div class="step"><h3>Step 2 · 筛除无效与背景落点</h3><p>基于Step 1原始eye_x/y、epr_level、head_z_mm及投影坐标，检查屏幕范围、层级、头距和WSI边界；再用NDPI低倍图的H&amp;E前景掩膜排除背景。掩膜使用RGB平均值&lt;245且R−G&gt;8或B−G&gt;8的染色信号，结合邻近暗组织及形态学处理。它是图像QC，不是细胞级诊断标注。</p>')
        parts.append(table(['判定','帧数'],[[esc(why(k)),v] for k,v in counts.items()]+[['保留',len(valid)]]))
        examples=[next(x for x in rows if x['exclusion_reason']==k) for k in counts]+[r]
        parts.append('<p>各类剔除的真实原始记录示例与通过记录：</p>'+raw_table(examples)+img(p/'coordinate_qc.png')+'</div>')
        parts.append('<div class="step"><h3>Step 3 · 归并同区域观察、连续缩放与回访</h3><p>输入为Step 2有效眼动坐标，以及由screen_x/y、screen_dimensions和downsample推导的显示窗口。以首个有效点为固定锚点；概览区域半径为全片1500×2600px。细览窗口短边阈值35000px；同窗口IoU≥0.65、回访≥0.75，中心平移≤参考边长20%；嵌套缩放覆盖≥0.85，倍率相对差≤2%视为同倍率。不是仅因距离近就无条件合并。无效帧不贡献支持，也不机械拆成新访问。</p><p>下表每一行都是归并后的候选访问；展开可直接看到参与的全部原始帧，间隔内缺失帧不会冒充有效记录。旧区域ID仅用于追溯，最终编号在Step 5生成。</p>')
        audits=[]
        for v in spatial['visits_audit']:
            ids=v['source_event_ids'];detail='<details><summary>'+str(len(ids))+'帧 · 查看原始数据</summary>'+raw_table([lookup[i] for i in ids])+'</details>'
            audits.append([v['visit_id'],v['region_id'],esc(v['kind']),f'{min(ids)}–{max(ids)}',esc(v['zoom_sequence']),detail])
        parts.append(table(['访问ID','旧区域ID','行为','原帧范围','相对原生倍率序列','数据来源'],audits)+'</div>')
        parts.append(f'<div class="step"><h3>Step 4 · 判断有效观察时间资格</h3><p>原timestamp中位间隔为{m["raw_median_interval"]:.6f}，头部FPS为{m["fps_header"]}（对应{1000/m["fps_header"]:g}ms/帧），两者单位关系尚未核实。因此主空间版不进行毫秒门槛筛选。另一版仅作为敏感性分析：假设每个有效帧支持下一帧间隔，末帧不增加时长，累计区间并集≥150ms才保留。不是“首末帧跨度×20ms”，无效间断不计时。</p>')
        parts.append(table(['访问ID','原始帧范围','有效帧数','估计支持ms（非实测）','150ms判断'],[[v['visit_id'],f"{min(v['source_event_ids'])}–{max(v['source_event_ids'])}",len(v['source_event_ids']),esc(v.get('valid_support')),('通过' if v.get('qualified') else '剔除：'+esc(v.get('exclusion_reason')))] for v in estimate['visits_audit']])+'</div>')
        parts.append('<div class="step"><h3>Step 5 · 压缩展示路径，重新连续编号</h3><p>输入为通过资格判断的访问及固定锚点。按1500×2600px归一化坐标，空间误差容忍1进行折线简化，保留端点与明显回返；存在可靠操作证据时保护对应访问，估计版另保护有效支持≥1000ms的访问。本数据没有鼠标时序，未启用操作保护。压缩不移动原始锚点，也不设目标点数。</p>')
        for title,res in [('主空间版',spatial),('50FPS估计150ms版',estimate)]:
            mapping={s['visit_id']:s for s in res['steps']}
            parts.append('<h4>'+title+'</h4>'+table(['原访问ID','处理','依据','最终访问 / 区域'],[[v['visit_id'],'保留' if v['retained'] else '压缩',esc(why(v['reason'])),f"{mapping[v['visit_id']]['step']:02d} / R{mapping[v['visit_id']]['region_id']:02d}" if v['visit_id'] in mapping else '不绘制，原记录仍保留'] for v in res['compression_audit']]))
        parts.append('</div><h3>对照效果 · 50FPS估计150ms版</h3><p>这是未校准时间假设下的结果，不是实测150ms注视轨迹。过滤会改变几何压缩路径，因此区域数量不必单调减少。</p>'+img(p/'fps_estimate_150ms/trajectory.png'))
        parts.append('<details><summary>本例完整逐帧数据 · 页面内检索核对</summary><input placeholder="查帧号、坐标或剔除原因" oninput="filterRows(this)">'+raw_table(rows)+'</details></section>')
    style='body{margin:0;background:#f0f4f7;color:#213d4c;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1440px;margin:auto;padding:32px}section{background:white;padding:28px;margin:24px 0;border-radius:12px}h1{font-size:34px}h2{border-bottom:3px solid #247098;padding-bottom:10px}h3{color:#17618a}.lead{font-size:21px}img{max-width:100%;height:auto;border:1px solid #dde5eb}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d8e1e7;text-align:left;vertical-align:top}th{background:#eaf1f5;white-space:nowrap}.scroll{overflow:auto}td details .scroll{max-width:950px;max-height:450px}details{padding:12px;background:#f6f8fa;margin:12px 0}summary{cursor:pointer;color:#17618a}.step{border-left:4px solid #87afc4;padding:4px 20px;margin:30px 0}input{padding:12px;width:90%}nav a{display:inline-block;margin:8px 20px 8px 0;color:#17618a}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PEAN专家眼动轨迹：结果与逐步处理</title><style>'+style+'</style><main><h1>PEAN专家眼动轨迹 · 结果与逐步处理</h1><p>独立PEAN报告。每个案例先看最终效果，再沿原始EPR字段 → 投影 → 质量筛选 → 观察归并 → 时间资格 → 展示压缩逐步核对。图片和所有核对表已嵌入本文件，无需逐个打开CSV。</p><nav>'+''.join(f'<a href="#{s["slide_id"]}">{s["slide_id"]}</a>' for s in summaries)+'</nav><section><h2>先读字段：原始数据与派生数据</h2>'+table(['字段','原始来源','用途'],[[esc(x) for x in r] for r in FIELDS])+'<p>屏幕投影尚未得到应用内容窗口边界验证；组织筛选不能完全替代视口外UI识别。NDPA只提供静态几何，TAIL提供病例级文本，没有点击/绘制时间，因此均不冒充鼠标事件。所有原始帧号从0开始，最终访问与区域编号从1开始。</p></section>'+''.join(parts)+'</main><script>function filterRows(input){let q=input.value.toLowerCase();input.parentElement.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}</script></html>'
    out=root/'PEAN结果报告.html';out.write_text(page,encoding='utf-8');print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
