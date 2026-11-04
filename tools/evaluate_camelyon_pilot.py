"""Join blinded Agent predictions to frozen official labels only after inference."""
import argparse,csv,html,json,math,subprocess,sys
from pathlib import Path

def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root
    labels=json.loads((root/'inputs/evaluation_labels.json').read_text());manifest=json.loads((root/'inputs/wsi_manifest.json').read_text())
    results={}
    for p in (root/'agent').glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('long_id') and 'pred_answer' in r:
            if r['long_id'] in results:raise ValueError('Duplicate inference result')
            results[r['long_id']]=(r,p)
    rows=[]
    for label in labels:
        key=label['long_id'];r,path=results.get(key,({},None));raw=r.get('pred_answer');answer={'benign':'Benign','malignant':'Malignant'}.get(str(raw).strip().lower().rstrip('.'))
        row={'case_id':label['case_id'],'slide_id':label['slide_id'],'truth':label['label'],'lesion_class':label['official']['class'],
             'prediction':answer,'raw_prediction':raw,'correct':answer==label['label'],'status':'valid' if answer else 'missing_or_invalid',
             'observations':len(r.get('trajectory',[])),'rounds':len(r.get('process',[])),'explanation':r.get('explanation'),'result_file':str(path) if path else None}
        rows.append(row)
        if path:
            assert r['ground_truth']=='WITHHELD'
            dest=root/('roi_'+label['case_id'])
            if not dest.exists():subprocess.run([sys.executable,str(Path(__file__).with_name('render_roi_navigation.py')),'--result',str(path),'--wsi',manifest[key]['path'],'--output',str(dest)],check=True)
    n=len(rows);correct=sum(r['correct'] for r in rows);pos=[r for r in rows if r['truth']=='Malignant'];neg=[r for r in rows if r['truth']=='Benign']
    tp=sum(r['prediction']=='Malignant' for r in pos);tn=sum(r['prediction']=='Benign' for r in neg)
    fp=sum(r['prediction']=='Malignant' for r in neg);fn=sum(r['prediction']=='Benign' for r in pos)
    metrics={'n':n,'correct':correct,'accuracy':correct/n,'accuracy_wilson95':wilson(correct,n),'sensitivity':tp/len(pos),'specificity':tn/len(neg),
             'balanced_accuracy':(tp/len(pos)+tn/len(neg))/2,'TP':tp,'TN':tn,'FP':fp,'FN':fn,
             'invalid_positive':sum(r['status']!='valid' for r in pos),'invalid_negative':sum(r['status']!='valid' for r in neg),
             'invalid_policy':'all selected cases remain in accuracy and sensitivity/specificity denominators',
             'scope':'six preselected development cases with 64 candidates each; not an unbiased population estimate',
             'task':'presence of lymph-node metastasis on this slide; not prognosis or future metastatic risk',
             'AUROC':None,'AUROC_reason':'no validated continuous slide-level risk scores',
             'new_cases_excluding_previous_C1':{'n':5,'correct':sum(r['correct'] for r in rows if r['case_id']!='C1')}}
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');(root/'case_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    with (root/'case_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=''.join(f'<tr><td>{r["case_id"]} / {r["slide_id"]}</td><td>{r["truth"]} ({r["lesion_class"]})</td><td>{r["prediction"] or "无有效输出"}</td><td>{"正确" if r["correct"] else "错误/失败"}</td><td>{r["observations"]}</td><td><a href="roi_{r["case_id"]}/index.html">ROI轨迹</a></td></tr>' for r in rows)
    cases=''.join(f'<section><h2>{r["case_id"]}：{r["prediction"] or "无有效输出"} / 标签 {r["truth"]}</h2><p>模型原始解释：{html.escape(str(r["explanation"]))}</p><a href="roi_{r["case_id"]}/roi_route_with_table.png"><img src="roi_{r["case_id"]}/roi_route_with_table.png"></a></section>' for r in rows if r['result_file'])
    (root/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CAMELYON16六例导航与分类</title><style>body{max-width:1500px;margin:35px auto;padding:0 24px;font:18px/1.8 sans-serif;color:#203d50;background:#f4f7fa}section{padding:25px;background:white;margin:24px 0}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd5dc;text-align:left}</style><h1>CAMELYON16：六例导航轨迹与转移癌分类</h1><p>Patho-R1-7B观察、Qwen3-4B执行、PLIP导航；直接读取原始WSI。每张64个图像组织筛选候选，初始3个观察，最多5轮。分类标签在推理结束后独立核对。</p>'+f'<p>正确 {correct}/{n}（{100*correct/n:.1f}%）；敏感度 {tp}/{len(pos)}；特异度 {tn}/{len(neg)}。这是预先固定的开发性小样本，包含此前使用过的C1。</p><table><tr><th>病例</th><th>官方标签</th><th>Agent分类</th><th>核对</th><th>观察数</th><th>轨迹</th></tr>'+table+'</table><p><a href="case_results.csv">逐例结果表</a> · <a href="metrics.json">指标与定义</a> · <a href="inputs/cohort.json">预先固定的病例与配置</a></p>'+cases+'</html>',encoding='utf-8')
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
