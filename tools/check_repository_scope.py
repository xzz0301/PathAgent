"""Reject tracked files outside the PathAgent/WSI-VQA baseline scope."""
import subprocess
from pathlib import PurePosixPath

ROOT_FILES = {'.gitignore', '.gitmodules', 'LICENSE', 'README.md', 'requirements.txt',
              'pathagent.py', 'navigation.py', 'trajectory.py', 'wsi_regions.py',
              'download_llms.py', 'download_checkpoints.ps1'}
DIRECTORIES = {'assets', 'configs', 'data_preparation_script', 'data_processing'}
MODELS = {'inference.py', 'input_validation.py', 'runtime.py'}
TESTS = {'test_consolidated.py', 'test_paper_navigation.py', 'test_wsi_regions.py',
         'test_wsi_vqa_metrics.py'}
DOCS = {'paper_alignment_wsi_vqa_20261104.md', 'quilt_20261031.md', 'repository_scope.md',
        'wsi_vqa_baseline_results.md'}
TOOLS = {'check_repository_scope.py', 'evaluate_wsi_vqa.py', 'prepare_wsi_vqa.py',
         'run_wsi_vqa.py', 'prepare_plip_features.py', 'quilt_acceptance.py',
         'render_roi_navigation.py', 'restore_patho_r1.py', 'setup_quilt.py', 'setup_wsi_metrics.py'}


def allowed(path):
    p=PurePosixPath(path)
    if len(p.parts)==1:return path in ROOT_FILES
    root=p.parts[0]
    if root in DIRECTORIES:return True
    if root=='external':return path=='external/PLIP'
    if root=='eval':return path=='eval/metics.py'
    if len(p.parts)!=2:return False
    return p.name in {'models':MODELS,'tests':TESTS,'docs':DOCS,'tools':TOOLS}.get(root,set())


def main():
    paths=[p for p in subprocess.check_output(['git','ls-files','-z']).decode('utf-8').split('\0') if p]
    rejected=[p for p in paths if not allowed(p)]
    if rejected:raise SystemExit('Out-of-scope tracked files:\n'+'\n'.join(rejected))
    print(f'Repository scope OK: {len(paths)} tracked paths')


if __name__=='__main__':main()
