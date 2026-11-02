# -*- coding: utf-8 -*-
"""Model verification: does the fast numba kernel reproduce the published BSM1 benchmarks?

Two independent checks, one per subcommand:

    python verify.py steady     closed-form steady state + derivative residual
    python verify.py xval [D]   dynamic cross-validation against bsm2-python for D days
                                (default 13.9, the full length of dryinfluent.csv)

'steady' is a 1-second self-test: it evaluates the analytic derivative at the published
open-loop steady state (data/asm1_values_ss_ps.csv) and reports the residual, which must be
zero if the ASM1 stoichiometry, the Takacs settler and the energy models are transcribed
correctly. It also reports the EQI and the three energy terms at that point.

'xval' is the real proof. It runs the official IWA reference implementation (the
bsm2-python package, scipy odeint, adaptive steps) and this kernel (fixed-step RK4, numba)
over the same influent sequence, from the same initial state, under the same open-loop
control, and prints the difference. Requires: pip install bsm2-python

Reference values from the last full run (D = 13.9 d, open loop):
    OFFICIAL  EQI=5810.8  AE=3341.4  PE=388.2  ME=240.0 kWh/d
    MINE      EQI=5779.5  AE=3341.4  PE=388.2  ME=240.0 kWh/d
    DELTA     EQI -31.3 (-0.5%)   effluent concentrations within 0.4 mg/L   ~18x faster
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bsm1_plant as P
import bsm1_fast as F
import env as E


def _load_reference_state():
    """Published open-loop steady state: 5 reactors followed by the 10-layer settler."""
    path = os.path.join(E._data_dir(), 'asm1_values_ss_ps.csv')
    lines = [l.strip() for l in open(path) if l.strip()]
    ss = [np.array([float(x) for x in l.split()]) for l in lines]
    return np.concatenate([np.vstack([ss[i] for i in range(5)]).ravel(), ss[7]]), ss


# ------------------------------------------------------------------ steady
def steady():
    y0, ss = _load_reference_state()
    react = np.vstack([ss[i] for i in range(5)])

    cin = np.zeros(21)
    for k, val in dict(SI=30.0, SS=69.5, XI=51.2, XS=202.32, XBH=28.17, XBA=0.0, XP=0.0,
                       SO=0.0, SNO=0.0, SNH=31.56, SND=6.95, XND=10.59, SALK=7.0,
                       TSS=211.27, Q=18446.0, TEMP=15.0).items():
        cin[getattr(P, k)] = val
    u = np.array([0., 0., 240., 240., 84., 3 * 18446., 18446., 385., 0.])

    # the clarifier feed carries the recycle flow; the internal recycle does NOT enter it
    y_feed = react[4].copy()
    y_feed[P.Q] = cin[P.Q] + u[6]
    ret, eff = F.settler_output_nb(y0[105:], y_feed, u[6], u[7])
    q = P.effluent_quantities(eff)
    print('steady state (open loop, KLa=[0,0,240,240,84], Qin=18446):')
    print('  Qeff=%.1f  SNH=%.3f SNO=%.3f NKJ=%.3f TN=%.3f TSS=%.3f COD=%.3f BOD5=%.3f'
          % (q['Q'], q['SNH'], q['SNO'], q['NKJ'], q['TN'], q['TSS'], q['COD'], q['BOD5']))
    print('  EQI = %.1f kg/d' % P.eqi(eff))
    ae, pe, me, tot = P.energy_total(u[0:5], u[5], u[6], u[7])
    print('  AE=%.1f kWh/d (%.2f kW)  PE=%.1f (%.2f kW)  ME=%.1f (%.2f kW)'
          % (ae, ae / 24, pe, pe / 24, me, me / 24))
    print('  Qin=%.1f Qw=%.1f  => Qeff should be %.1f' % (cin[P.Q], u[7], cin[P.Q] - u[7]))

    d = F.plant_deriv_nb(y0, 0.0, cin, u)
    print('\nderivative residual (must be 0 for a true steady state):')
    print('  max|d(reactor, non-SO)|=%.4f  max|d(settler)|=%.4f'
          % (np.abs(np.delete(d[:105].reshape(5, 21), P.SO, axis=1)).max(), np.abs(d[105:]).max()))


# ------------------------------------------------------------------ xval
def xval(days=13.9):
    try:
        os.environ['TQDM_DISABLE'] = '1'
        from bsm2_python.bsm1_ol import BSM1OL
    except ImportError:
        raise SystemExit('this check needs the reference implementation: pip install bsm2-python')

    inf = np.loadtxt(os.path.join(E._data_dir(), 'dryinfluent.csv'), delimiter=',')
    out = []

    t0 = time.time()
    bsm = BSM1OL(data_in=inf, timestep=1 / 1440, endtime=days, evaltime=None,
                 tempmodel=False, activate=False)
    bsm.stabilize()
    tstab = time.time() - t0
    t1 = time.time()
    bsm.simulate(plot=False)
    tsim = time.time() - t1
    nsteps = len(bsm.simtime)
    eff = bsm.ys_eff_all[-1]
    off = dict(EQI=bsm.eqi_all[-1], EQI_mean=bsm.eqi_all[-1440:].mean(),
               AE=bsm.ae * 24, PE=bsm.pe * 24, ME=bsm.me * 24,
               Q=eff[P.Q], SNH=eff[P.SNH], SNO=eff[P.SNO], TSS=eff[P.TSS])
    out.append('OFFICIAL: stabilize %.1fs  simulate %.1fs  (%d steps, %.0f steps/s)'
               % (tstab, tsim, nsteps, nsteps / max(tsim, 1e-9)))
    out.append('  EQI=%.1f (day-mean %.1f)  AE=%.1f PE=%.1f ME=%.1f kWh/d'
               % (off['EQI'], off['EQI_mean'], off['AE'], off['PE'], off['ME']))
    out.append('  eff Q=%.1f SNH=%.3f SNO=%.3f TSS=%.3f' % (off['Q'], off['SNH'], off['SNO'], off['TSS']))

    # --- this kernel: same influent, same initial state, same open-loop control ---
    y, _ = _load_reference_state()
    u = np.array([0., 0., 240., 240., 84., 3 * 18446., 18446., 385., 0.])
    dt = 1.0 / 1440.0
    N = int(round(days / dt))
    ae_s = pe_s = me_s = 0.0
    eqi_s = 0.0
    t2 = time.time()
    for k in range(N):
        j = max(min(np.searchsorted(inf[:, 0], k * dt, side='right') - 1, len(inf) - 1), 0)
        inf_t = inf[j, 1:]
        y = F.rk4_step(y, dt, inf_t, u, 6)
        if k >= N - 1440:
            eff2, ret2, me, ae, pe = F.settle_metrics(y, inf_t, u)
            ae_s += ae
            pe_s += pe
            me_s += me
            eqi_s += F.eqi_nb(eff2)
    tmine = time.time() - t2
    out.append('MINE    : simulate %.1fs  (%d steps, %.0f steps/s)  [%.1fx faster]'
               % (tmine, N, N / tmine, tsim / max(tmine, 1e-9)))
    out.append('  EQI=%.1f (day-mean %.1f)  AE=%.1f PE=%.1f ME=%.1f kWh/d'
               % (F.eqi_nb(eff2), eqi_s / 1440, ae_s / 1440, pe_s / 1440, me_s / 1440))
    out.append('  eff Q=%.1f SNH=%.3f SNO=%.3f TSS=%.3f'
               % (eff2[P.Q], eff2[P.SNH], eff2[P.SNO], eff2[P.TSS]))
    out.append('DELTA: EQI %+.1f  SNH %+.4f  SNO %+.4f  TSS %+.4f  AE %+.2f  PE %+.2f  ME %+.2f'
               % (F.eqi_nb(eff2) - off['EQI'], eff2[P.SNH] - off['SNH'], eff2[P.SNO] - off['SNO'],
                  eff2[P.TSS] - off['TSS'], ae_s / 1440 - off['AE'], pe_s / 1440 - off['PE'],
                  me_s / 1440 - off['ME']))

    text = '\n'.join(out)
    print(text)
    open(os.path.join(HERE, '_verify_result.txt'), 'w', encoding='utf-8').write(text + '\n')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'steady'
    if what == 'steady':
        steady()
    elif what == 'xval':
        xval(float(sys.argv[2]) if len(sys.argv) > 2 else 13.9)
    else:
        raise SystemExit('unknown check %r; choose steady or xval' % what)
