# D:\PathCode\download_checkpoints.ps1
#
# 把 PathAgent 需要的三个模型权重下载到 D:\PathCode\checkpoints\
#
# 用法（默认只下零门槛的，不下需要审批的）：
#   powershell -ExecutionPolicy Bypass -File D:\PathCode\download_checkpoints.ps1
#   powershell -ExecutionPolicy Bypass -File D:\PathCode\download_checkpoints.ps1 -PlipOnly
#   powershell -ExecutionPolicy Bypass -File D:\PathCode\download_checkpoints.ps1 -IncludeGated
#
# 前置条件：
#   本机 huggingface.co 直连不通，脚本会自动走 hf-mirror.com 镜像
#   需要审批的模型（Patho-R1）必须先在自己账号里点过同意，并设置 HF_TOKEN

[CmdletBinding()]
param(
    [switch]$PlipOnly,        # 只检查/下载 PLIP
    [switch]$IncludeGated,    # 连同需要审批的 Patho-R1 一起下（需先设 HF_TOKEN）
    [string]$Root = 'D:\PathCode\checkpoints'
)

$ErrorActionPreference = 'Stop'

# --- 环境准备 ---
$env:HF_ENDPOINT = 'https://hf-mirror.com'   # 本机 huggingface.co 不通，必须走镜像
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$py = 'D:\PathCode\.venv310\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "找不到 Python 解释器: $py" }
New-Item -ItemType Directory -Path $Root -Force | Out-Null

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' PathAgent 权重下载' -ForegroundColor Cyan
Write-Host " 目标目录: $Root" -ForegroundColor Cyan
Write-Host ' 镜像: hf-mirror.com' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan

# --- 模型清单 ---
# gated = $false 的可以直接下；$true 的需要先在 HF 网页上点同意
$models = @(
    [pscustomobject]@{
        Dir   = 'plip'
        Repo  = 'vinid/plip'
        Size  = '约 0.58 GB'
        Gated = $false
        Role  = 'Navigator（用问题检索 patch）—— 论文原版权重'
        Arg   = '--plip_ckpt'
    },
    [pscustomobject]@{
        Dir   = 'qwen-executor'
        Repo  = 'Qwen/Qwen3-4B'
        Size  = '约 7.5 GB（4bit 约 1.9 GB）'
        Gated = $false
        Role  = 'Executor（纯文本决策员）—— ★ 论文原文指定的就是 Qwen3-4B'
        Arg   = '--qwen_ckpt'
    },
    [pscustomobject]@{
        Dir   = 'patho-r1-7b'
        Repo  = 'WenchuanZhang/Patho-R1-7B'
        Size  = '约 15.5 GB（4bit 约 3.9 GB）'
        Gated = $true
        Role  = 'Perceptor（看图写描述）—— ★ README 明确写 PATHOR1_7B，需 HF 审批'
        Arg   = '--patho_r1_ckpt'
    },
    [pscustomobject]@{
        Dir   = 'qwen-vl-3b'
        Repo  = 'Qwen/Qwen2.5-VL-3B-Instruct'
        Size  = '约 7.0 GB（4bit 约 1.8 GB）'
        Gated = $false
        Role  = 'Perceptor 的零门槛替代（架构与 Patho-R1 完全相同，但没有病理微调）'
        Arg   = '--patho_r1_ckpt（替代）'
    }
)

# --- 筛选要下载的 ---
$todo = $models | Where-Object {
    if ($PlipOnly) { $_.Dir -eq 'plip' }
    elseif ($IncludeGated) { $true }
    else { -not $_.Gated }
}

Write-Host ''
Write-Host '本次计划处理:' -ForegroundColor Yellow
foreach ($m in $todo) {
    Write-Host ("  [{0,-16}] {1,-34} {2}" -f $m.Dir, $m.Repo, $m.Size)
}
Write-Host ''

# --- 逐个下载 ---
$failed = @()
foreach ($m in $todo) {
    $dest = Join-Path $Root $m.Dir
    $marker = Join-Path $dest '.download-complete'

    Write-Host ("── {0}  ({1})" -f $m.Repo, $m.Size) -ForegroundColor Green
    Write-Host ("   用途: {0}" -f $m.Role) -ForegroundColor DarkGray
    Write-Host ("   参数: {0} {1}" -f $m.Arg, $dest) -ForegroundColor DarkGray

    if (Test-Path $marker) {
        Write-Host '   已存在，跳过（要重下就删掉 .download-complete）' -ForegroundColor DarkGray
        continue
    }

    if ($m.Gated -and -not $env:HF_TOKEN) {
        Write-Host '   ⏭ 跳过：该模型需要审批，且未设置 HF_TOKEN' -ForegroundColor Yellow
        Write-Host '      步骤：① 打开 https://huggingface.co/{0} 点同意 → ② 建 token → ③ $env:HF_TOKEN="hf_xxx"' -f $m.Repo -ForegroundColor Yellow
        continue
    }

    try {
        & $py -c @"
from huggingface_hub import snapshot_download
snapshot_download(r'''$($m.Repo)''', local_dir=r'''$dest''')
"@
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Path $marker -Force | Out-Null
            $sz = [math]::Round(((Get-ChildItem $dest -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
            Write-Host ("   ✅ 完成，落地 {0} GB" -f $sz) -ForegroundColor Green
        } else {
            throw "python 退出码 $LASTEXITCODE"
        }
    } catch {
        Write-Host ("   ❌ 失败: {0}" -f $_.Exception.Message.Split([char]10)[0]) -ForegroundColor Red
        $failed += $m.Repo
    }
    Write-Host ''
}

# --- 汇总 ---
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 现有权重总览' -ForegroundColor Cyan
Get-ChildItem $Root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $sz = [math]::Round(((Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
    Write-Host ("  {0,-18} {1,6} GB" -f $_.Name, $sz)
}

if ($failed.Count) {
    Write-Host ''
    Write-Host ("以下仓库下载失败，可重跑本脚本：{0}" -f ($failed -join ', ')) -ForegroundColor Yellow
}

Write-Host ''
Write-Host '下一步：把上面的路径填进 pathagent.py 的参数里' -ForegroundColor Cyan
