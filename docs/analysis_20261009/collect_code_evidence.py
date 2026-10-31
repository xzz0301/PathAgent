"""Read-only project audit; writes evidence only alongside this script. No model/network calls."""
import ast
import csv
import difflib
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r'D:\PathCode')
OUT = Path(__file__).resolve().parent
EXPERT = Path(r'C:\Users\14910\Documents\ChatGPT\病理医学研究\trajectory_demo')


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def inventory(root, paths):
    rows = []
    for p in sorted(paths):
        source = p.read_text(encoding='utf-8-sig')
        tree = ast.parse(source, filename=str(p))
        functions = []

        def walk(body, prefix=''):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    calls = sorted({ast.unparse(c.func) for c in ast.walk(n) if isinstance(c, ast.Call)})
                    functions.append({'name': prefix+n.name, 'line': n.lineno, 'end_line': n.end_lineno,
                                      'signature': ast.unparse(n.args), 'calls': calls,
                                      'returns': sorted({ast.unparse(r.value) for r in ast.walk(n)
                                                         if isinstance(r, ast.Return) and r.value is not None}),
                                      'doc': ast.get_docstring(n) or ''})
                    walk(n.body, prefix+n.name+'.')
                elif isinstance(n, ast.ClassDef):
                    walk(n.body, prefix+n.name+'.')

        walk(tree.body)
        rows.append({'path': str(p), 'relative_path': p.relative_to(root).as_posix(),
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'lines': len(source.splitlines()),
                     'functions': functions,
                     'imports': [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]})
    return rows


def dataset_summary(base, case):
    patches = base/'patches'/case
    features = base/'features'/case
    names = sorted(p.name for p in patches.glob('*.png'))
    desc = read_json(base/'descriptions.json').get(case, {})
    arrays = [(p.name, np.load(p, allow_pickle=False)) for p in sorted(features.glob('*.npy'))]
    norms = [float(np.linalg.norm(a)) for _, a in arrays]
    return {'base': str(base), 'case': case, 'patch_count': len(names),
            'description_count': len(desc), 'empty_description_count': sum(not v.strip() for v in desc.values()),
            'feature_count': len(arrays), 'feature_shapes': dict(Counter(str(a.shape) for _, a in arrays)),
            'finite_features': all(bool(np.isfinite(a).all()) for _, a in arrays),
            'feature_norm_range': [min(norms), max(norms)] if norms else [],
            'missing_patch_files': sorted(set(desc)-set(names)),
            'missing_features': sorted(set(Path(n).stem for n in desc)-set(Path(n).stem for n, _ in arrays)),
            'questions': read_json(base/'questions.json')}


def result_summary():
    results = []
    for p in sorted((ROOT/'demo'/'results').rglob('*.json')):
        data = read_json(p)
        if not isinstance(data, dict) or not {'long_id', 'pred_answer', 'question'} <= set(data):
            continue
        points = data.get('trajectory', [])
        events = p.with_name(p.stem+'_events.jsonl')
        counts = Counter()
        if events.exists():
            counts.update(json.loads(line)['event'] for line in events.read_text(encoding='utf-8').splitlines() if line.strip())
        results.append({'path': str(p), 'case': data['long_id'], 'pred_answer': data['pred_answer'],
                        'ground_truth': data.get('ground_truth'), 'observations': len(points),
                        'statuses': dict(Counter(x.get('status', 'missing') for x in points)),
                        'rounds': sorted({x.get('round') for x in points}),
                        'modes': [x.get('mode') for x in data.get('process', [])],
                        'trajectory_files': data.get('trajectory_files'), 'geometry': data.get('geometry'),
                        'event_counts': dict(counts), 'runtime': data.get('runtime'),
                        'first_points': [{k:x.get(k) for k in ('step', 'patch_name', 'x','y','width','height','magnification')} for x in points[:8]]})
    return results


def focused_probes():
    # Extract pure functions via AST to avoid importing heavyweight models/argparse entry points.
    eval_tree = ast.parse((ROOT/'eval'/'metics.py').read_text(encoding='utf-8'))
    node = next(n for n in eval_tree.body if isinstance(n, ast.FunctionDef) and n.name=='acc_of_seq')
    scope = {'difflib': difflib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<acc_of_seq>', 'exec'), scope)
    sys.path.insert(0, str(ROOT))
    from demo.trajectory_viz import TraceRecorder, export_trajectory_artifacts, split_zoom_patch
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)/'100_200.png'
        Image.new('RGB', (512,512), 'pink').save(p)
        trace = TraceRecorder(folder, 'probe')
        trace.start_observation(round=1, action='probe', patch_name=p.name, x=100,y=200,width=512,height=512,magnification=10)
        files = export_trajectory_artifacts(trace.points, folder, folder,'case','probe',
                                            {'coordinate_scale':4,'coordinate_space':'test','level0_verified':False})
        exported = read_json(Path(files['json']))['trajectory']
        probe = {'unfinished_export': {'input_status':'started','export_count':len(exported),
                                       'export_status':exported[0]['status']},
                 'crop_20x_from_10x': {'source_pixels':[512,512], 'crop_pixels':list(split_zoom_patch(p,20,10,4)[0][0].size),
                                       'adds_new_native_pixels':False},
                 'mcq_unknown': {'choices':['Benign','Malignant'],'ground_truth':'Malignant','prediction':'Unknown',
                                 'reported_correct':scope['acc_of_seq'](['Benign','Malignant'],'Malignant','Unknown')},
                 'mcq_empty': {'reported_correct':scope['acc_of_seq'](['Benign','Malignant'],'Malignant','')}}
    for n in (6, 64, 100, 1000):
        probe.setdefault('selection_budget', []).append({'candidates':n,'default_initial':max(1,int(n*.1)),
                                                        'perceptor_observation_cap':5,'default_replenish':max(1,int(n*.05))})
    return probe


def main():
    primary = [p for p in ROOT.glob('*.py')]
    for d in ('models','data_processing','data_preparation_script','eval','demo','pathogaze'):
        primary.extend(p for p in (ROOT/d).rglob('*.py') if '__pycache__' not in p.parts)
    primary.append(ROOT/'external'/'PLIP'/'plip.py')
    inv = inventory(ROOT, primary)
    expert_inv = inventory(EXPERT, list(EXPERT.glob('*.py'))+list((EXPERT/'tests').glob('*.py')))
    evidence = {'analysis_date':'2026-10-09 Asia/Shanghai',
                'git': {'head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                        'tracked_diff':subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','HEAD'],text=True).splitlines(),
                        'remote':subprocess.check_output(['git','-C',str(ROOT),'remote','-v'],text=True).strip()},
                'datasets':[dataset_summary(ROOT/'demo','DX1_SICAP001'),
                            dataset_summary(ROOT/'demo'/'c1_independent','DX1_C1')],
                'results':result_summary(), 'probes':focused_probes(), 'syntax_files':len(inv)+len(expert_inv)}
    for p in (ROOT/'demo'/'c1_independent'/'candidate_manifest.json', EXPERT/'runs'/'video_audit_20261009'/'C1_paired_comparison.json'):
        if p.exists():
            value=read_json(p)
            if 'patches' in value:
                value={k:v for k,v in value.items() if k not in ('patches','wsi_description')}
            evidence.setdefault('other_evidence', {})[str(p)]=value
    for filename,value in [('audit_evidence.json',evidence),('function_inventory.json',{'project':inv,'expert_side':expert_inv})]:
        (OUT/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    diff=''.join(difflib.unified_diff((ROOT/'pathagent.py').read_text(encoding='utf-8').splitlines(True),
                                     (ROOT/'pathagent_demo.py').read_text(encoding='utf-8').splitlines(True),
                                     fromfile='pathagent.py (tracked baseline)',tofile='pathagent_demo.py (local extension)'))
    (OUT/'pathagent_demo_vs_original.diff').write_text(diff,encoding='utf-8')
    lines=['# 函数与调用索引', '', '生成于 2026-10-09。AST 静态提取，不执行推理。调用名集合不是实际运行顺序；主文档解释语义和流程。',
           '', '外部 PLIP 仅索引主入口 plip.py，其训练/复现实验目录不属于当前 Agent 调用链。', '']
    for title,items in [('D:\\PathCode 主项目',inv),('C盘专家轨迹关联代码',expert_inv)]:
        lines.extend(['## '+title,''])
        for entry in items:
            lines.extend(['### '+entry['relative_path'],'',f"源文件：[打开]({entry['path'].replace(chr(92),'/')})；{entry['lines']} 行；SHA256：`{entry['sha256']}`。",''])
            if not entry['functions']:
                lines.extend(['该文件以模块顶层脚本运行，没有函数定义。',''])
            for f in entry['functions']:
                lines.extend([f"- **`{f['name']}({f['signature']})`** — [第 {f['line']} 行]({entry['path'].replace(chr(92),'/')}:{f['line']})。"])
                if f['doc']:
                    lines.append('  - 源注释：'+f['doc'].splitlines()[0].strip())
                lines.append('  - 静态调用：'+('; '.join('`'+x+'`' for x in f['calls']) or '无'))
                if f['returns']:
                    lines.append('  - 返回表达式：'+('; '.join('`'+x.replace(chr(10),' ')[:400]+'`' for x in f['returns']) or '无'))
            lines.append('')
    (OUT/'函数与调用索引.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
