param([string]$OutputDir = '', [int]$MinimumRounds = 1, [string]$Config = '', [string]$Python = '', [string]$PerceptorCheckpoint = '', [int]$InitialSamples = 5, [int]$MaxAttempts = 3, [int]$PerceptorTokens = 512)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if (-not $Config) {$Config=Join-Path $taskRoot 'configs/local_8gb.json'}
if (-not $Python) {$Python=Join-Path $taskRoot '.venv310/Scripts/python.exe'}
if (-not $OutputDir) {$OutputDir=Join-Path $taskRoot ('demo/results/c1_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))}
$inputDir=Join-Path $taskRoot 'demo\c1_independent'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONUTF8='1'
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$runArgs=@('-u', (Join-Path $taskRoot 'pathagent.py'), '--config', $Config,
 '--descriptions_file', (Join-Path $inputDir 'descriptions.json'),
 '--questions_file', (Join-Path $inputDir 'questions.json'),
 '--feature_dir', (Join-Path $inputDir 'features'),
 '--patch_root', (Join-Path $inputDir 'patches'),
 '--geometry_file', (Join-Path $inputDir 'geometry.json'),
 '--save_dir', $OutputDir, '--dataset_name','wsi_vqa','--seed','128',
 '--initial_sample_count', [string]$InitialSamples,'--max_attempts', [string]$MaxAttempts,'--perceptor_max_new_tokens', [string]$PerceptorTokens,
 '--min_observation_rounds', [string]$MinimumRounds)
if ($PerceptorCheckpoint) {$runArgs+=@('--patho_r1_ckpt', $PerceptorCheckpoint)}
& $Python @runArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'run.log')
if($LASTEXITCODE -ne 0){throw "C1 demo failed: $LASTEXITCODE"}
