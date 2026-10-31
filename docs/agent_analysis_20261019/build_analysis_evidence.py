"""Regenerate source excerpts and evidence for the 2026-10-19 analysis."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(r"D:\PathCode")
OUT = Path(__file__).resolve().parent
SERVER = Path(r"C:\Users\14910\Documents\ChatGPT\病理医学研究\server_preparation\20261015")
EXPERT = Path(r"C:\Users\14910\Documents\ChatGPT\病理医学研究\trajectory_demo")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "lines": len(read(path).splitlines()),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    excerpts = [
        ("初选、候选集合、逐块观察与状态构造", "pathagent.py", 171, 236),
        ("充分分支：最终整合并退出", "pathagent.py", 239, 266),
        ("zoom分支：父块/子块检索与终结", "pathagent.py", 268, 366),
        ("缺失信息补检与预算耗尽后的回答", "pathagent.py", 368, 425),
        ("JSON生成调用的基础封装", "models/inference.py", 10, 78),
        ("Executor：A/B/C判断链", "models/inference.py", 80, 200),
        ("最终答案：累计描述转为答案", "models/inference.py", 202, 308),
        ("Perceptor：图像与任务转为描述", "models/inference.py", 310, 392),
        ("描述摘要：控制累计证据的组织方式", "models/inference.py", 394, 496),
        ("状态文本：倍率、坐标和描述", "data_processing/utils.py", 107, 131),
        ("原版zoom：已有图块的裁剪", "data_processing/utils.py", 146, 203),
    ]
    chunks = ["# 原版 Agent 核心源码摘录\n",
              f"原版提交：`{git('rev-parse', 'HEAD')}`。以下仅摘录原版文件，保留源码原文及源文件行号范围；代码块内未加入行号。各段控制流讲解见[原版分析](D:/PathCode/docs/agent_analysis_20261019/01_原版PathAgent代码与论文对应分析.md)。\n"]
    for i, (title, rel, lo, hi) in enumerate(excerpts, 1):
        path = ROOT / rel
        lines = read(path).splitlines()
        assert 1 <= lo <= hi <= len(lines), (path, lo, hi)
        chunks.extend([f"## {i}. {title}\n",
                       f"来源：[{rel}:{lo}]({path.as_posix()}:{lo})，原文件第{lo}—{hi}行。\n",
                       "```python\n" + "\n".join(lines[lo-1:hi]) + "\n```\n"])
    (OUT / "原版Agent核心源码摘录.md").write_text("\n".join(chunks), encoding="utf-8")

    original = read(ROOT / "pathagent.py").splitlines(keepends=False)
    local = read(ROOT / "pathagent_demo.py").splitlines(keepends=False)
    diff = difflib.unified_diff([x+"\n" for x in original], [x+"\n" for x in local],
                              fromfile="original/pathagent.py", tofile="local/pathagent_demo.py")
    (OUT / "pathagent_demo_vs_original.diff").write_text("".join(diff), encoding="utf-8")
    server_file = SERVER / "bundle/pathagent_server.py"
    server_lines = read(server_file).splitlines()
    diff = difflib.unified_diff([x+"\n" for x in local], [x+"\n" for x in server_lines],
                              fromfile="local/pathagent_demo.py", tofile="server/pathagent_server.py")
    (OUT / "pathagent_server_vs_demo.diff").write_text("".join(diff), encoding="utf-8")

    original_files = [ROOT / p for p in ["pathagent.py", "models/inference.py", "data_processing/utils.py", "eval/metics.py"]]
    original_files += sorted((ROOT / "data_preparation_script").glob("*.py"))
    local_files = [ROOT / "pathagent_demo.py"] + sorted((ROOT / "demo").glob("*.py"))
    index = ["# 函数与输入输出阅读索引\n",
             "本索引从2026-10-19当前源码的AST自动提取函数签名与所在行。主分析中的表格解释输入输出语义；本文件用于快速定位。原版与本地扩展分开排列。\n"]
    syntax_files = []
    for label, paths in [("原版源码", original_files), ("本地扩展", local_files)]:
        index.append(f"## {label}\n")
        for path in paths:
            tree = ast.parse(read(path), filename=str(path))
            syntax_files.append(str(path))
            index.append(f"### [{path.name}]({path.as_posix()})\n")
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs = [(node.name, node)]
                elif isinstance(node, ast.ClassDef):
                    funcs = [(f"{node.name}.{child.name}", child) for child in node.body
                             if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
                else:
                    continue
                for name, func in funcs:
                    signature = ast.unparse(func.args)
                    index.append(f"- [`{name}({signature})`]({path.as_posix()}:{func.lineno})：第{func.lineno}—{func.end_lineno}行。\n")
            if not any(isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) for n in tree.body):
                index.append("该脚本在模块顶层执行参数解析与处理。\n")
    (OUT / "函数定位索引.md").write_text("\n".join(index), encoding="utf-8")

    models = []
    for name in ["Qwen--Qwen3-4B.api.json", "WenchuanZhang--Patho-R1-7B.api.json", "vinid--plip.api.json", "wisdomik--Quilt-Llava-v1.5-7b.api.json"]:
        source = SERVER / name
        data = json.loads(read(source))
        record = {"id": data.get("id", data.get("modelId")), "revision": data.get("sha"),
                  "metadata_snapshot_date": "2026-10-15", "source_file": str(source),
                  "safetensors": data.get("safetensors"),
                  "architectures": data.get("config", {}).get("architectures")}
        params = (data.get("safetensors") or {}).get("total")
        if params:
            record["bf16_weight_bytes"] = params * 2
            record["bf16_weight_gib"] = params * 2 / 2**30
        models.append(record)
    (OUT / "sources/model_parameter_evidence.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")

    # Evaluate only the pure scoring function; do not import optional metric/model dependencies.
    score_tree = ast.parse(read(ROOT / "eval/metics.py"))
    score_func = next(n for n in score_tree.body if isinstance(n, ast.FunctionDef) and n.name == "acc_of_seq")
    scope = {"difflib": difflib}
    exec(compile(ast.Module(body=[score_func], type_ignores=[]), "acc_of_seq_excerpt", "exec"), scope)
    empty_prediction = scope["acc_of_seq"](["Benign", "Malignant"], "Malignant", "")

    features = []
    import numpy as np
    for rel in ["demo/features/DX1_SICAP001", "demo/c1_independent/features/DX1_C1"]:
        paths = sorted((ROOT / rel).glob("*.npy"))
        norms = [float(np.linalg.norm(np.load(p))) for p in paths]
        features.append({"directory": str(ROOT / rel), "count": len(paths),
                         "l2_min": min(norms) if norms else None, "l2_max": max(norms) if norms else None})
    results = []
    for rel in ["demo/results/c1_independent_20261009/DX1_C1_3032b9e5.json",
                "demo/results/c1_coverage_guard_20261009/DX1_C1_3032b9e5.json"]:
        data = json.loads(read(ROOT / rel))
        points = data.get("trajectory", [])
        results.append({"path": str(ROOT / rel), "pred_answer": data.get("pred_answer"),
                        "ground_truth": data.get("ground_truth"), "trajectory_count": len(points),
                        "completed_count": sum(p.get("status") == "completed" for p in points),
                        "actions": sorted({str(p.get("action")) for p in points})})

    supporting_files = [ROOT / "README.md", ROOT / "requirements.txt", ROOT / "external/PLIP/plip.py",
                        ROOT / "checkpoints/Qwen3-4B/config.json", ROOT / "checkpoints/plip/config.json"]
    supporting_files += sorted((ROOT / "data_preparation_script").glob("*.sh"))
    supporting_files += [EXPERT / name for name in ["pathagent_trace.py", "run_dual_v2.py", "simplify_v2.py",
                         "gaze_observation_v4.py", "run_gaze_observation_v4.py", "trajectory_demo.py",
                         "render_corrected_outputs.py"]]
    files = list(dict.fromkeys(original_files + local_files + [server_file] + supporting_files))
    manifest = {
        "analysis_date": "2026-10-19", "root": str(ROOT), "git_head": git("rev-parse", "HEAD"),
        "tracked_diff_name_status": git("diff", "HEAD", "--name-status"),
        "source_files": [file_record(p) for p in files],
        "syntax_checked_files": syntax_files,
        "paper": {"version": "arXiv:2511.17052v1", "url": "https://arxiv.org/html/2511.17052v1",
                  "saved_html_sha256": hashlib.sha256((OUT / "sources/PathAgent_arxiv_2511.17052.html").read_bytes()).hexdigest()},
        "models": models,
        "resource_formulas": {"gib_bytes": 2**30, "bf16_bytes_per_parameter": 2,
                              "qwen3_kv_bytes_per_token_batch1_bf16": 2*36*8*128*2,
                              "qwen3_8192_tokens_kv_gib": 2*36*8*128*2*8192/2**30,
                              "runtime_ranges": "planning estimates, not measured on the two-4090 server"},
        "feature_norms_measured": features,
        "existing_c1_results_read": results,
        "pure_function_check": {"acc_of_seq_empty_prediction_returns": empty_prediction},
        "hardware": {"user_provided": "2 x RTX 4090, assumed 24GB per GPU", "server_executed": False},
        "diff_format": "Both files normalized to newline-terminated lines for readable unified diff; originals unchanged.",
    }
    (OUT / "代码核对清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"excerpt_sections": len(excerpts), "syntax_checked_files": len(syntax_files),
                      "tracked_diff": manifest["tracked_diff_name_status"], "features": features,
                      "existing_c1_results": results, "empty_prediction_returns": empty_prediction}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
