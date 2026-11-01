from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for script in ['01_run_reference_cases.py','02_run_verification.py','03_plot_results.py']:
    print(f'\n=== {script} ===')
    subprocess.check_call([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT)
print('\nAll reference FE and verification tasks completed.')
