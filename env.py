# -*- coding: utf-8 -*-
"""Deployment environment: the validated plant model driven by a supervisory policy.

The upper layer chooses, once per 15-minute decision interval, two setpoints that are
handed to the plant's own PI loops:
    a = [SO_sp, SNO2_sp]
SO_sp is applied to reactors 3, 4 and 5 (the steady-state map shows that the three
aerobic reactors share one optimum) and SNO2_sp is the nitrate setpoint of reactor 2,
which drives the internal recycle.

The plant underneath is the cross-validated BSM1 + two-step-nitrification + N2O model,
integrated at one-minute resolution with the official PI gains.

OBJECTIVES
    water quality   EQI [kg/d]        IWA BSM1 effluent quality index
    energy          kWh/d            aeration + pumping + mixing
    carbon          kgCO2e/d         indirect (dynamic grid factor) + direct N2O/CH4

SAFETY
    NH4-N <= 4, total N <= 18, TSS <= 30 g/m3 (BSM1 limits). Violations are counted as
    accumulated time, which is the metric the competition asks for.
"""
import numpy as np
import loop as L
import n2o as N
import twin as TW
from bsm1_fast import SO, SNO, SNH, TSS, Q, eqi_nb
from control_arms import KLA_INIT, DT

DEC = 15                       # decision interval: 15 min = 15 one-minute steps
NSUB = 4
LIM_SNH = 4.0
LIM_TN = 18.0
LIM_TSS = 30.0
SP_LO = np.array([0.6, 0.25])
SP_HI = np.array([4.0, 5.0])
NACT = 2
NIN = 13

# normalisation constants (the BSM1 default operating point, from the steady-state map)
EQI_REF = 5318.0                # 10-day dynamic mean at the BSM1 default setpoints
KWH_REF = 3637.0
CO2_REF = 7900.0
WARMUP_D = 2.0                  # un-scored start-up phase (identical for every arm)
PEN_VIOL = 5.0                 # reward penalty per hour of effluent-limit violation
W_EQI, W_KWH, W_CO2 = 0.5, 0.5, 0.35


def arm_weights(arm):
    """Reward weights of each arm; the only difference between B/C and D is the carbon term."""
    if arm == 'B':
        return dict(eqi=0.5, kwh=0.5, co2=0.0)
    if arm == 'C':
        return dict(eqi=0.5, kwh=0.5, co2=0.0)
    return dict(eqi=0.4, kwh=0.25, co2=0.35)      # arm D: three objectives


class Env:
    def __init__(self, inf_arr, ef_arr, twin_obj, ref_tn, days, seed=0):
        self.inf = inf_arr
        self.ef = ef_arr
        self.tw = twin_obj
        self.ref_tn = ref_tn
        self.rng = np.random.RandomState(seed)
        self.reset(days)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, days, y0=None):
        self.days = days
        self.nstep = int(days / DT)
        self.t = 0
        self.st = L.fresh_state()
        self.y = np.load('_ss_A_n2o.npy') if y0 is None else y0.copy()
        self.acc = np.zeros(L.NACC)
        self.sp = np.array([2.0, 1.0])
        self.f_load = 1.0
        self.hist = []
        self.warm_steps = int(WARMUP_D / DT)
        self.acc0 = None                       # snapshot taken at the end of warm-up
        self.acc0_step = 0
        return self.obs()

    def obs(self):
        yr = self.y[:105].reshape(5, 21)
        k = min(self.t, self.inf.shape[0] - 1)
        inf = self.inf[k]
        self.f_load = (inf[Q] * N.influent_tn(inf)) / self.ref_tn
        return np.array([
            yr[2][SO], yr[3][SO], yr[4][SO], yr[1][SNO],
            self.sp[0], self.sp[1],
            inf[Q] / 21477.0, inf[SNH] / 30.0, inf[TSS] / 236.0, inf[15] / 15.0,
            TW.tou_price(self.t * DT) / 1.05,
            TW.grid_factor(self.t * DT) / TW.EF_BASE,
            self.f_load,
        ])

    def apply(self, action):
        """Clip the requested setpoints into the admissible box and run one interval."""
        sp = np.clip(np.asarray(action, dtype=float), SP_LO, SP_HI)
        self.sp = sp
        n = min(DEC, self.nstep - self.t)
        if n <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, True
        a0 = self.acc.copy()
        y, ctrl, xs, xa, xq, ma, mq, acc = L.cl_steps(
            self.y, self.inf[self.t:self.t + n], self.ef[self.t:self.t + n], n, DT, NSUB,
            self.st[0], self.st[1], self.st[2], self.st[3], np.array([sp[0], sp[0], sp[0], sp[1]]),
            3, KLA_INIT, self.st[4], self.st[5], self.st[6], self.st[7], self.st[8], self.acc)
        self.y = y
        self.st = (self.st[0], self.st[1], self.st[2], self.st[3], xs, xa, xq, ma, mq)
        self.acc = acc
        self.t += n
        if self.t >= self.warm_steps and self.acc0 is None:
            # the start-up transient ends here: discard everything accumulated so far
            self.acc0 = acc.copy()
            self.acc0_step = self.t
            a0 = acc.copy()
        d = acc - a0
        dt_days = n * DT
        d_kwh = d[0] + d[1] + d[2]
        d_co2 = d[3] + d[4] / 1000.0 * (44.0 / 28.0) * N.GWP_N2O + d[5] / 1000.0 * (16.0 / 14.0) * N.GWP_CH4
        d_viol = (d[12] + d[13] + d[14]) * 24.0            # hours
        eff, ret, me, ae, pe = L.settle_metrics(y[:225], self.inf[self.t - 1], ctrl)
        st, efx, wa, pr, a, b, c = N.n2o_fluxes(y, self.inf[self.t - 1], ctrl)
        info = dict(snh=float(eff[SNH]), sno=float(eff[SNO]), tss=float(eff[TSS]),
                    eqi=float(eqi_nb(eff)), kwh=float(ae + pe + me),
                    kla3=float(ctrl[2]), kla4=float(ctrl[3]), kla5=float(ctrl[4]),
                    qintr=float(ctrl[5]), sp=sp.copy(), viol_h=d_viol,
                    kwh_i=float(d_kwh), eqi_i=float(d[7]), co2_i=float(d_co2),
                    days_i=float(dt_days))
        self.hist.append(info)
        return d_kwh, d[7], d_co2, d_viol, dt_days, self.t >= self.nstep

    def reward(self, arm, kwh_i, eqi_i, co2_i, viol_h, dt_days):
        w = arm_weights(arm)
        r = -(w['eqi'] * (eqi_i / dt_days) / EQI_REF
              + w['kwh'] * (kwh_i / dt_days) / KWH_REF
              + w['co2'] * (co2_i / dt_days) / CO2_REF)
        return float(r - PEN_VIOL * viol_h / 24.0)

    # ------------------------------------------------------------------ outcomes
    def outcome(self):
        a = self.acc - (self.acc0 if self.acc0 is not None else 0.0)
        days = (self.acc[11] - self.acc0_step) * DT
        vol = a[6]
        kwh_d = (a[0] + a[1] + a[2]) / days
        ind = a[3] / days
        n2o_co2 = a[4] / 1000.0 * (44.0 / 28.0) * N.GWP_N2O / days
        ch4_co2 = a[5] / 1000.0 * (16.0 / 14.0) * N.GWP_CH4 / days
        return dict(days=days, kwh_d=kwh_d, ind_d=ind,
                    n2o_co2_d=n2o_co2, ch4_co2_d=ch4_co2,
                    tot_d=ind + n2o_co2 + ch4_co2, eqi_d=a[7] / days,
                    kwh_m3=kwh_d / (vol / days), vol_m3=vol, vol_d=vol / days,
                    snh_mean=a[8] / days, sno_mean=a[9] / days, tss_mean=a[10] / days,
                    viol_snh=a[12] / days, viol_tn=a[13] / days, viol_tss=a[14] / days,
                    tn_mean=a[15] / days, org_n=a[15] / days - a[8] / days - a[9] / days,
                    aer_kwh=a[0] / days, pump_kwh=a[1] / days, mix_kwh=a[2] / days,
                    n2o_gN_d=a[4] / days)


def _data_dir():
    """Locate the official IWA BSM influent tables.

    Preference order: ./data next to this file (the layout shipped with the source
    supplement), then the bsm2_python package data directory. This keeps the package
    runnable on a machine that only has the public reference implementation installed.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, 'data')
    if os.path.isfile(os.path.join(local, 'dryinfluent.csv')):
        return local
    try:
        import bsm2_python, os as _os
        d = _os.path.join(_os.path.dirname(bsm2_python.__file__), 'data')
        if _os.path.isfile(_os.path.join(d, 'dryinfluent.csv')):
            return d
    except Exception:
        pass
    raise FileNotFoundError(
        'dryinfluent.csv not found: expected ./data/dryinfluent.csv next to env.py, or an '
        'installed bsm2-python package (pip install bsm2-python).')


def build_inputs(days, mode='dry'):
    """Influent table (nstep x 21) and dynamic grid factor (nstep) for a campaign.

    mode='dry'  cycles the official BSM1 dry-weather influent file;
    mode='rain' cycles the official BSM1 rain-weather file, which contains the same dry
    pattern interrupted by the standard rain events.
    """
    import csv, os
    D = _data_dir()
    name = 'dryinfluent.csv' if mode == 'dry' else 'raininfluent.csv'
    rows = []
    with open(os.path.join(D, name)) as f:
        for r in csv.reader(f):
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                continue                      # two rows of the rain file are corrupted
    base = np.array(rows)[:, 1:]
    if mode == 'rain':
        # The official rain file stores flow in thousands of m3/d (21.477 = 21477 m3/d);
        # the dry file stores it in m3/d. Without this the plant is starved by 1000x.
        base[:, Q] *= 1000.0
    nstep = int(days / DT)
    out = np.array([base[i % base.shape[0]] for i in range(nstep)])
    ef = np.array([TW.grid_factor(i * DT) for i in range(nstep)])
    return out, ef


def ref_tn_load():
    import csv, os
    D = _data_dir()
    with open(os.path.join(D, 'dryinfluent.csv')) as f:
        first = [float(x) for x in next(csv.reader(f))][1:]
    inf0 = np.array(first)
    return N.influent_tn(inf0) * inf0[Q]


def make_twin():
    tw = TW.Twin('_sweep_fine.csv', np.arange(0.6, 4.01, 0.2), np.arange(0.25, 5.01, 0.25))
    tw.ref_tn = ref_tn_load()
    return tw
