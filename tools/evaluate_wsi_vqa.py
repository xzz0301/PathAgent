"""Aggregate frozen WSI-VQA test results and render only three preset cases."""
import argparse
import difflib
import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_processing.utils import make_unique_id
from tools.prepare_wsi_vqa import save


def compatible_correct(prediction, truth, choices):
    if not prediction:
        return False
    target = difflib.SequenceMatcher(None, prediction, truth).quick_ratio()
    return not any(difflib.SequenceMatcher(None,prediction,choice).quick_ratio()>target for choice in choices)


def norm(value):
    return re.sub(r'\s+', ' ', str(value).strip().lower().rstrip('.'))


def collect_rows(reference, by_case, results, exclusions):
    rows=[];opened=[];closed=[]
    for index,q in enumerate(reference):
        key=by_case.get(q['Id'])
        result,path=results.get((key,q['Question']),({},None)) if key else ({},None)
        if result and result['ground_truth']!='WITHHELD':
            raise ValueError('Inference labels were not blinded')
        pred=str(result.get('pred_answer') or '').strip()
        row={'index':index,'case_id':q['Id'],'question':q['Question'],'truth':q['Answer'],'prediction':pred,
             'choices':q.get('Choice'),'result_file':str(path) if path else None,
             'status':'returned' if pred else ('uncalibrated' if q['Id'] in exclusions else 'missing'),
             'calibration_eligible':q['Id'] not in exclusions,
             'exclusion_reason':exclusions.get(q['Id']),
             'rounds':len(result.get('process',[])),'observations':len(result.get('trajectory',[])),
             'termination_reason':result.get('termination_reason')}
        rows.append(row)
        if q.get('Choice'):
            row['compatible_correct']=compatible_correct(pred,q['Answer'],q['Choice'])
            row['strict_correct']=bool(pred) and norm(pred)==norm(q['Answer'])
            row['exact_option']=norm(pred) in {norm(v) for v in q['Choice']}
            closed.append(row)
        else:
            opened.append(row)
    return rows,closed,opened


def require_eligible_coverage(rows):
    missing=[r['index'] for r in rows if r['calibration_eligible'] and not r['prediction']]
    if missing:
        raise ValueError(f'Incomplete calibrated evaluation: {len(missing)} unanswered eligible questions; '
                         f'reference indices={missing}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    reference=json.loads((root/'inputs/reference.json').read_text())
    manifest=json.loads((root/'inputs/wsi_manifest.json').read_text())
    by_case={m['slide_id']:key for key,m in manifest.items()}
    excluded_path=root/'inputs/excluded_slides.json'
    exclusions={r['case_id']:r['reason'] for r in json.loads(excluded_path.read_text())} if excluded_path.exists() else {}
    results={}
    for f in (root/'agent').glob('*.json'):
        value=json.loads(f.read_text())
        if value.get('long_id') and 'pred_answer' in value:
            results[(value['long_id'],value['question'])]=(value,f)
    rows,closed,opened=collect_rows(reference,by_case,results,exclusions)
    assert len(rows)==735 and len(closed)==390 and len(opened)==345
    require_eligible_coverage(rows)
    save(root/'case_results.json',rows)
    eligible_closed=[r for r in closed if r['calibration_eligible']]
    eligible_open=[r for r in opened if r['calibration_eligible']]
    eligible_rows=[r for r in rows if r['calibration_eligible']]
    metrics={'n_wsi':86,'n_questions':735,'closed_n':390,'open_n':345,'missing':sum(not r['prediction'] for r in rows),
             'calibrated_wsi':len(manifest),'excluded_slides':exclusions,
             'eligible_questions':len(eligible_rows),'eligible_closed_n':len(eligible_closed),'eligible_open_n':len(eligible_open),
             'closed_compatible_accuracy_pct':100*sum(r['compatible_correct'] for r in closed)/390,
             'closed_strict_accuracy_pct':100*sum(r['strict_correct'] for r in closed)/390,
             'closed_nonexact_option':sum(not r['exact_option'] for r in closed),
             'missing_denominator_policy':'All 735 fixed questions retained; missing closed answers incorrect.',
             'compatible_rule':'Upstream SequenceMatcher quick_ratio strict-greater comparison; empty outputs explicitly incorrect.',
             'paper_closed_reference_pct':56.32,'paper_protocol':'390 closed / 345 open; open metrics on open subset only',
             'method':'paper-aligned reconstruction with regional plans and post-zoom reassessment, not unmodified upstream',
             'mean_evaluation_rounds':sum(r['rounds'] for r in rows)/735,
             'source_question_sha256':hashlib.sha256((root/'inputs/reference.json').read_bytes()).hexdigest()}
    metrics['calibrated_subset']={
        'closed_compatible_accuracy_pct':100*sum(r['compatible_correct'] for r in eligible_closed)/len(eligible_closed),
        'closed_strict_accuracy_pct':100*sum(r['strict_correct'] for r in eligible_closed)/len(eligible_closed),
        'closed_denominator':len(eligible_closed),'open_denominator':len(eligible_open),
        'missing':sum(not r['prediction'] for r in eligible_rows),
        'mean_evaluation_rounds':sum(r['rounds'] for r in eligible_rows)/len(eligible_rows)}
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    def score_open(group):
        gts={str(r['index']):[r['truth']] for r in group};res={str(r['index']):[r['prediction']] for r in group}
        meteor=Meteor()
        try:
            bleu,_=Bleu(4).compute_score(gts,res)
            met,_=meteor.compute_score(gts,res)
            rouge,_=Rouge().compute_score(gts,res)
        finally:
            proc=getattr(meteor,'meteor_p',None)
            if proc is not None and proc.poll() is None:
                proc.kill();proc.wait()
        return {**{f'BLEU_{i+1}':100*v for i,v in enumerate(bleu)},'METEOR':100*met,'ROUGE_L':100*rouge}
    metrics['open_metrics_pct']=score_open(opened)
    metrics['calibrated_subset']['open_metrics_pct']=score_open(eligible_open)
    save(root/'metrics.json',metrics)
    esc=html.escape
    parts=['<!doctype html><meta charset="utf-8"><title>WSI-VQA复现统计与案例</title><style>body{max-width:1450px;margin:30px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243b50}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ccd5df;text-align:left}th{background:#edf3f8}.cards{display:flex;flex-wrap:wrap;gap:15px}.card{width:300px;border:1px solid #d8e0e8;padding:12px}.card img{width:100%}pre{white-space:pre-wrap;font:14px/1.6 system-ui}section{margin-top:40px;border-top:2px solid #bacbd8;padding-top:20px}</style><h1>WSI-VQA：统计与三个固定案例</h1>']
    parts.append('<p>86张WSI、735问答；5倍全组织候选，无64个上限；初选ceil10%、补选ceil5%、Top5专项观察、最多5轮。PLIP、Patho-R1-7B、Qwen3-4B；Quilt非量化通用描述。采用论文图例的逐区域计划与高倍后反思，歧义选择和预处理参数详见protocol.json与Git改动说明。</p>')
    if exclusions:
        parts.append('<p><b>校准范围：</b>缺少扫描倍率的'+esc(', '.join(exclusions))+'未运行，也未猜测倍率。'
                     f'实际纳入{len(manifest)}张WSI、{len(eligible_rows)}题；'
                     '原735题完整统计保留未运行题（闭题计错，开放题为空输出），并另列实际校准子集结果，二者不混用。</p>')
    parts.append('<table><tr><th>统计项</th><th>本次结果</th></tr>'+''.join('<tr><td>'+esc(k)+'</td><td>'+esc(str(v))+'</td></tr>' for k,v in metrics.items())+'</table>')
    chosen=['TCGA-BH-A202','TCGA-D8-A27M','TCGA-LD-A74U']
    examples=[]
    for case in chosen:
        row=next(r for r in rows if r['case_id']==case)
        if not row['result_file']:
            parts.append('<section><h2>'+case+'</h2><p>预定案例无有效结果，保留失败状态。</p></section>');continue
        result=json.loads(Path(row['result_file']).read_text());examples.append({'case':case,'question_index':row['index']})
        figure=root/'selected_cases'/case
        if not (figure/'roi_route_with_table.png').exists():
            subprocess.run([sys.executable,str(Path(__file__).with_name('render_roi_navigation.py')),
                            '--result',row['result_file'],'--wsi',manifest[by_case[case]]['path'],
                            '--output',str(figure)],check=True)
        parts.append('<section><h2>'+case+'</h2><p>问题：'+esc(row['question'])+'</p><p>选项：'+esc(str(row['choices']))+'</p><p>模型结果：'+esc(row['prediction'])+'；参考答案（推理后独立核对）：'+esc(row['truth'])+'</p><p>最终解释：'+esc(result['explanation'])+'</p>')
        parts.append('<img style="width:100%" src="selected_cases/'+case+'/roi_route_with_table.png">')
        for stage in result['process']:
            decision=stage['evaluation_result'];round_id=stage['attempt']
            parts.append(f'<h3>第{round_id}轮</h3><div class="cards">')
            observations=[r for r in result['trajectory'] if r['round']==round_id]
            for obs in observations:
                relative=Path(obs['input_image_path']).relative_to(root).as_posix()
                text=obs['description'];answer=re.search(r'<answer>(.*?)</answer>',text,re.S)
                summary=answer.group(1).strip() if answer else text
                parts.append('<div class="card"><img src="'+relative+'"><p>'+esc(obs['patch_name'])+f' · {obs["magnification"]:g}倍</p><p>实际WSI范围：'+esc(str(obs['roi']))+'</p><pre>'+esc(summary)+'</pre></div>')
            parts.append('</div><p>初步预测：'+esc(str(decision.get('answer')))+'；充分性：'+esc(str(decision.get('sufficient')))+'；动作：'+esc(stage['mode'])+'</p>')
            if decision.get('missing_info'):
                parts.append('<p>缺失信息：'+esc(str(decision['missing_info']))+'</p>')
            if 'validated_region_plans' in stage:
                parts.append('<pre>实际采用的逐ROI计划：'+esc(json.dumps(stage['validated_region_plans'],ensure_ascii=False,indent=2))+'</pre>')
            parts.append('<details><summary>本轮全部证据来源与原始描述</summary><pre>'+esc(json.dumps(stage['state'],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<p>终止原因：'+esc(result['termination_reason'])+'</p></section>')
    parts.append('<p>案例在推理前固定为三张WSI的首个测试问题，没有按正确与否挑选。未触发的步骤不补写。完整机器记录保留用于复核，本页面仅展示统计与三例。</p>')
    save(root/'selected_examples.json',examples)
    (root/'WSI-VQA结果报告.html').write_text(''.join(parts),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
