import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contracts import scaled_input,validate_case,select_tags,sampled_metrics

CASE = dict(ramp1_C_min=2.0, hold1_T_C=108.0, hold1_d_min=65.0,
            ramp2_C_min=2.0, hold2_T_C=176.0, hold2_d_min=108.0,
            hbot_W_m2K=70.0,htop_W_m2K=95.0,Lt_m=0.04,Lc_m=0.03,total_process_time_s=18800.)
EXP = dict(r1_min=1.5,r1_max=3,ht1_min=100,ht1_max=120,hd1_min=50,hd1_max=70,
           r2_min=1.5,r2_max=3,ht2_min=175,ht2_max=185,hd2_min=100,hd2_max=110,
           hcb_min=40,hcb_max=90,hct_min=70,hct_max=120,lt_min=0.02,lt_max=0.07,
           T_ini=20,t_max=18800,T_max=250,l_max=0.05,h_max=120)


class ContractTests(unittest.TestCase):
    def test_selected_reference_timing(self):
        timing = validate_case(CASE,EXP,{'part_len':0.03})
        self.assertAlmostEqual(timing['heating_hold_end_s']/60,251)
        self.assertAlmostEqual(timing['cooling_s']/60,62+1/3)

    def test_fixed_horizon_cannot_be_changed(self):
        with self.assertRaises(ValueError):
            validate_case(dict(CASE,total_process_time_s=12000),EXP,{'part_len':0.03})

    def test_thickness_cannot_be_changed(self):
        with self.assertRaises(ValueError):
            validate_case(dict(CASE,Lc_m=0.02),EXP,{'part_len':0.03})

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_case(dict(CASE,ramp1_C_min=float('nan')),EXP,{'part_len':0.03})

    def test_scaling_recovers_physical_case(self):
        u = scaled_input(CASE,EXP)
        self.assertEqual(len(u),9)
        self.assertAlmostEqual(u[0]*250/18800*60,2.0)
        self.assertAlmostEqual(u[1]*250,108)
        self.assertAlmostEqual(u[2]*18800/60,65)
        self.assertAlmostEqual(u[6]*120,70)
        self.assertAlmostEqual(u[8]*0.05,0.04)

    def test_missing_checkpoint_partner_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'xpidon_class_10000_T.pkl').write_bytes(b'not loaded')
            with self.assertRaises(FileNotFoundError):
                select_tags(temp,['10000'])

    def test_incomplete_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            select_tags('.', ['05000'])

    def test_metrics_use_thickness_not_midpoint(self):
        metrics = sampled_metrics([0,10],[0,0.5,1],[[20,20,20],[210,190,200]],
                                  [[0.05,0.05,0.05],[0.88,0.94,0.91]])
        self.assertEqual(metrics['sampled_peak_part_temperature_C'],210)
        self.assertEqual(metrics['sampled_final_alpha_min'],0.88)
        self.assertAlmostEqual(metrics['sampled_final_alpha_spread'],0.06)

    def test_nan_field_is_rejected(self):
        with self.assertRaises(ValueError):
            sampled_metrics([0,10],[0,1],[[20,20],[200,float('nan')]],[[0.05,0.05],[0.9,0.9]])


if __name__=='__main__':
    unittest.main()
