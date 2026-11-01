from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CureCycle:
    T0_C: float
    ramp1_C_min: float
    hold1_T_C: float
    hold1_d_min: float
    ramp2_C_min: float
    hold2_T_C: float
    hold2_d_min: float
    total_process_time_s: float
    final_temperature_C: float = 20.0

    @classmethod
    def from_case(cls, case: dict, config: dict):
        return cls(
            T0_C=float(config['initial_conditions']['temperature_C']),
            ramp1_C_min=float(case['ramp1_C_min']),
            hold1_T_C=float(case['hold1_T_C']),
            hold1_d_min=float(case['hold1_d_min']),
            ramp2_C_min=float(case['ramp2_C_min']),
            hold2_T_C=float(case['hold2_T_C']),
            hold2_d_min=float(case['hold2_d_min']),
            total_process_time_s=float(case.get('total_process_time_s', config['cure_cycle']['total_process_time_s'])),
            final_temperature_C=float(config['cure_cycle']['final_temperature_C']),
        )

    @property
    def breakpoints_min(self):
        ramp1_end = (self.hold1_T_C - self.T0_C) / self.ramp1_C_min
        hold1_end = ramp1_end + self.hold1_d_min
        ramp2_end = hold1_end + (self.hold2_T_C - self.hold1_T_C) / self.ramp2_C_min
        hold2_end = ramp2_end + self.hold2_d_min
        process_end = self.total_process_time_s / 60.0
        if process_end <= hold2_end:
            raise ValueError(
                f'Total process time {process_end:.3f} min must exceed hold-2 end {hold2_end:.3f} min.'
            )
        return ramp1_end, hold1_end, ramp2_end, hold2_end, process_end

    @property
    def cooling_rate_C_min(self):
        *_, hold2_end, process_end = self.breakpoints_min
        return (self.hold2_T_C - self.final_temperature_C) / (process_end - hold2_end)

    def temperature_C(self, t_s):
        scalar = np.isscalar(t_s)
        t_min = np.atleast_1d(np.asarray(t_s, dtype=float)) / 60.0
        r1_end, h1_end, r2_end, h2_end, t_end = self.breakpoints_min
        cool_rate = self.cooling_rate_C_min

        Ta = np.full_like(t_min, self.final_temperature_C, dtype=float)
        m1 = t_min <= r1_end
        Ta[m1] = self.T0_C + self.ramp1_C_min * t_min[m1]
        m2 = (t_min > r1_end) & (t_min <= h1_end)
        Ta[m2] = self.hold1_T_C
        m3 = (t_min > h1_end) & (t_min <= r2_end)
        Ta[m3] = self.hold1_T_C + self.ramp2_C_min * (t_min[m3] - h1_end)
        m4 = (t_min > r2_end) & (t_min <= h2_end)
        Ta[m4] = self.hold2_T_C
        m5 = (t_min > h2_end) & (t_min <= t_end)
        Ta[m5] = self.hold2_T_C - cool_rate * (t_min[m5] - h2_end)
        Ta = np.maximum(Ta, self.final_temperature_C)
        return float(Ta[0]) if scalar else Ta

    def sensor_table(self, case_id: str, count: int = 100) -> pd.DataFrame:
        t = np.linspace(0.0, self.total_process_time_s, count)
        Ta = self.temperature_C(t)
        return pd.DataFrame({
            'case_id': case_id,
            'sensor_id': np.arange(count),
            't_s': t,
            't_min': t / 60.0,
            't_norm_0_1': t / self.total_process_time_s,
            'Ta_C': Ta,
        })
