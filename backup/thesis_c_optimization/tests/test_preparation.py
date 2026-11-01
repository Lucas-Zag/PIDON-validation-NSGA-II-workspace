import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cycle import cycle_timing
from prepare import ROOT, inspect


class PreparationTests(unittest.TestCase):
    def test_baseline_physical_units(self):
        case = dict(r1=2.2, ht1=110, hd1=58, r2=2.2, ht2=180, hd2=105)
        timing = cycle_timing(case, 20, 18800)
        self.assertAlmostEqual(timing['heating_hold_end_s'], 14143.636363636364)
        self.assertAlmostEqual(timing['cooling_s'], 4656.363636363636)
        self.assertTrue(timing['positive_cooling_interval'])

    def test_impossible_cooling_is_rejected(self):
        timing = cycle_timing(dict(r1=1.5, ht1=120, hd1=70, r2=1.5, ht2=185, hd2=150), 20, 18800)
        self.assertFalse(timing['positive_cooling_interval'])

    def test_nonphysical_inputs(self):
        with self.assertRaises(ValueError):
            cycle_timing(dict(r1=0, ht1=110, hd1=58, r2=2, ht2=180, hd2=105), 20, 18800)

    def test_inspection_does_not_modify_siblings(self):
        config = json.loads((ROOT / 'config/project.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            for name in config['siblings'].values():
                (workspace/name).mkdir()
            xp = workspace/config['siblings']['xpidon']
            params = {k+s: value for k, pair in zip(('r1','ht1','hd1','r2','ht2','hd2'),
                ((1.5,3),(100,120),(50,70),(1.5,3),(175,185),(100,110)))
                for s, value in zip(('_min','_max'),pair)}
            params.update(T_ini=20, t_max=18800, T_max=250)
            (xp/'exp_params.json').write_text(json.dumps(params))
            comp = workspace/config['siblings']['comparison']
            (comp/'run_fe_cases.py').write_text('from fe_reference.solver import solve_case\n')
            def snapshot():
                return {str(f.relative_to(workspace)):hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in workspace.rglob('*') if f.is_file()}
            before = snapshot()
            report = inspect(workspace,config)
            self.assertEqual(before,snapshot())
            self.assertEqual(report['projects']['comparison']['call_evidence'][0]['kinds'], ['python'])
            self.assertEqual(report['projects']['xpidon']['invalid_box_corners'], [])


if __name__ == '__main__':
    unittest.main()
