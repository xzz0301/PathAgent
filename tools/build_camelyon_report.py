"""Build a separate, self-contained Agent result report from verified pilot outputs."""
import argparse,base64,html,json
from pathlib import Path
from PIL import Image

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def esc(x):return html.escape(str(x))
def picture(p):
    with Image.open(p) as im:im.verify()
    return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def main(root):
    rows=load(root/'case_results.json');m=load(root/'metrics.json');pair=load(root/'C1固定ROI回放/paired_results.json')
    assert len(rows)==m['n']==6 and sum(r['correct'] for r in rows)==m['correct']
    parts=['<h1>PathAgent：六例原始WSI导航与分类结果</h1><p>2026-11-04 · 6000D服务器 · Patho-R1-7B观察、Qwen3-4B执行、PLIP导航；候选通用描述沿用Quilt NF4。</p>',
           '<section><h2>结果：6例全部预测阳性，3例阴性均误报</h2><p class="lead">正确3/6（50%） · 敏感度3/3（100%） · 特异度0/3（0%）</p><p>准确率Wilson 95%区间为18.8%–81.2%；平衡准确率50%。无无效输出。排除此前开发中使用过的C1后，新增5例中正确2例。本试验是预先固定的开发性小样本，不是总体性能估计；全判阳性无法支持有效区分阴阳性。</p>',
           table(['切片','官方标签','病灶类型','Agent预测','判定','观察次数'],[[r['case_id']+' / '+r['slide_id'],r['truth'],r['lesion_class'],r['prediction'],'正确' if r['correct'] else '假阳性',r['observations']] for r in rows]),
           '<p>任务是判断该淋巴结切片有无转移癌，不是未来转移风险或预后。阳性切片分类正确也不能证明所浏览ROI含有病灶，仍需病灶标注或专家核查。</p></section>',
           '<section><h2>本轮实际执行过程</h2><p>原始WSI → 64个图像组织候选 → Quilt区域描述与PLIP特征 → 初始3次Patho-R1观察 → Qwen3判断是否继续导航 → 最终分类 → 与隔离保存的官方标签核对。</p><p>C1复用既有64候选；其余在组织覆盖≥25%的网格中等间隔选择64个，未使用专家眼动或病灶坐标选区。最多5轮，实际六例均在第1轮结束，C1额外观察1个放大ROI，其他病例仅3个初始ROI。当前结果主要暴露有限观察下的分类问题，尚未形成充分的多轮导航性能验证。</p><p>问题答案统一为WITHHELD；评价脚本在推理结束后才合并官方reference。Patho-R1与Qwen3均为BF16原权重。正式轨迹只展示实际观察ROI框和访问顺序，不用框中心冒充精确注视点。</p></section>']
    for r in rows:
        c=r['case_id'];data=load(root/'agent'/('DX1_'+c+'_3032b9e5.json'))
        assert data['ground_truth']=='WITHHELD' and data['pred_answer']==r['prediction']
        assert data['runtime']['image_backend']=='wsi'
        parts.append('<section><h2>'+c+' · '+('分类正确' if r['correct'] else '假阳性')+'</h2>'+picture(root/('roi_'+c)/'roi_route_with_table.png')+'<p><b>最终解释（模型原文）：</b>'+esc(r['explanation'])+'</p><details><summary>逐次ROI观察、原始描述与真实读取信息</summary>')
        for t in data['trajectory']:
            a=t['acquisition'];roi=t['roi'];assert a['image_source']=='original_wsi' and roi==a['roi']
            parts.append('<h3>观察 '+str(t['step'])+'</h3>'+table(['倍率','level-0矩形(x,y,w,h)','读取层 / downsample','输入尺寸'],[[t['magnification'],[roi[k] for k in ['x0','y0','width0','height0']],str(a['read_level'])+' / '+str(a['actual_downsample']),a['output_size']]])+'<p>'+esc(t['description'])+'</p>')
        parts.append('</details></section>')
    parts.append('<section><h2>C1固定ROI对照：真正读取高分辨率像素</h2><p>保持4个ROI位置不变，旧图和新WSI图分别使用相同Patho-R1、问题和种子回放；原基线SHA未改变。第4次放大观察的1024×1024 level-0组织范围，新输入512×512、处理器504×504；旧输入256×256、处理器252×252。</p>'+picture(root/'C1固定ROI回放/paired_inputs.png')+table(['观察','旧图尺寸','WSI新图尺寸','旧处理器尺寸','新处理器尺寸'],[[x['step'],x['old_size'],x['new_size'],x['old_model_input']['processed_sizes'],x['wsi_model_input']['processed_sizes']] for x in pair])+'<p>四次新输入定位均未满足当前完整坐标协议（0/4），没有接受精确关注点；因此本轮仍使用ROI框。回放只验证输入与观察，不产生新的整片分类。</p></section>')
    parts.append('<section><h2>发现的问题与下一步核查</h2><p>已确认的问题：三例阴性全被判为阳性；六例均早期结束；只覆盖64候选中的3–4次观察。当前不能把流畅的文字解释视为已证实的形态证据。</p><p>下一步应逐层核对阴性切片：原始ROI图像是否正确 → Patho-R1描述是否把正常淋巴组织当成异型上皮 → Qwen3是否忽略阴性描述或过早停止 → 候选选区是否有偏。需要从保存图像和原始描述定位误差来源，再讨论提示词或停止规则；本轮没有为提高分数修改策略或重跑挑选结果。</p><p>报告中的ROI是实际模型观察区域。专家同片轨迹一致性与病灶覆盖评价尚未在这六例中完成；没有经过验证的连续风险分数，因此不报告AUROC。</p></section>')
    style='body{max-width:1500px;margin:32px auto;padding:0 24px;background:#f2f5f8;color:#203d50;font:18px/1.8 system-ui}section{background:white;padding:28px;margin:24px 0}img{max-width:100%;height:auto}td,th{padding:12px;border-bottom:1px solid #ccd5dc;text-align:left;vertical-align:top}table{border-collapse:collapse;width:100%}.scroll{overflow:auto}.lead{font-size:24px;color:#963c26}summary{cursor:pointer;background:#edf2f6;padding:12px}details{margin:18px 0}'
    (root/'PathAgent结果报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PathAgent六例结果报告</title><style>'+style+'</style>'+''.join(parts)+'</html>',encoding='utf-8')
    print('REPORT_COMPLETE',root/'PathAgent结果报告.html')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
