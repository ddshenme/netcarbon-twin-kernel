# -*- coding: utf-8 -*-
"""Steady-state operating map of the closed loop: (DO setpoint, nitrate setpoint) -> objectives.

For every (dissolved-oxygen setpoint, nitrate setpoint) pair the plant is integrated to a new
steady state under the official PI control layer and the three objectives are recorded:

    water quality   EQI (kg pollution units/d), effluent NH4-N, NO3-N, TSS
    energy          aeration + pumping + mixing (kWh/d) and its indirect CO2
    carbon          direct N2O (kgCO2e/d) from the two-step nitrification overlay

This map is the digital twin's steady-state predictor (twin.py reads the CSV it writes), and it
is what locates the carbon-optimal window and the Pareto front between the three objectives.

Three grids, one code path
--------------------------
    python sweep.py coarse    11 x 4   points, 30 d each   -> _sweep.csv
    python sweep.py fine      18 x 12  points, 40 d each   -> _sweep_fine.csv   (default)
    python sweep.py ext        9 x 5   points, 40 d each   -> _sweep_ext.csv
    python sweep.py all       all three, in the order above

'fine' is the grid twin.py and figures.py consume; 'ext' extends it towards the high
internal-recycle setpoints that carry the low-carbon end of the Pareto front. Cost is roughly
2.8 s per point on 16 cores, so 'fine' takes about 10 minutes.
"""
import os, sys, time, csv, itertools
import numpy as np
import loop as L
import n2o as N
from bsm1_fast import SO, SNO, SNH, TSS, Q, eqi_nb
from control_arms import KLA_INIT, DT
import env as E

# name -> (DO setpoints, nitrate setpoints, days per point, output csv, print the table?)
GRIDS = {
    'coarse': ([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5],
               [0.5, 1.0, 1.5, 2.0], 30.0, '_sweep.csv', False),
    'fine':   ([round(0.6 + 0.2 * i, 2) for i in range(18)],
               [round(0.25 + 0.25 * i, 2) for i in range(12)], 40.0, '_sweep_fine.csv', False),
    'ext':    ([0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0],
               [3.0, 3.5, 4.0, 4.5, 5.0], 40.0, '_sweep_ext.csv', True),
}


def reference_influent():
    """Constant influent of the reference steady state: first row of the official dry-weather table."""
    rows = []
    with open(os.path.join(E._data_dir(), 'dryinfluent.csv')) as f:
        for i, r in enumerate(csv.reader(f)):
            rows.append([float(x) for x in r])
            if i >= 2:
                break
    inf0 = np.array(rows[0][1:])
    return inf0, N.influent_tn(inf0) * inf0[Q]


def run_grid(name, y0, inf0, tn_load):
    """Sweep one named grid and return its records."""
    sps, snos, days, out_csv, verbose = GRIDS[name]
    recs = []
    t0 = time.time()
    for so_sp, sno_sp in itertools.product(sps, snos):
        st = L.fresh_state()
        sp = np.array([so_sp, so_sp, so_sp, sno_sp])
        r = L.cl_epoch(y0, inf0, days, DT, 4, st[0], st[1], st[2], st[3], sp, 3, KLA_INIT,
                       st[4], st[5], st[6], st[7], st[8])
        y, ctrl = r[0], r[1]
        eff, ret, me, ae, pe = L.metrics(y, inf0, ctrl)
        ind, n2o, ch4, pr = L.carbon(y, inf0, ctrl)
        yr = y[:105].reshape(5, 21)
        recs.append(dict(so=so_sp, sno=sno_sp,
                         do3=yr[2][SO], do4=yr[3][SO], do5=yr[4][SO],
                         snh=eff[SNH], snoe=eff[SNO], tss=eff[TSS], eqi=eqi_nb(eff),
                         ae=ae, pe=pe, me=me, kwh=ae + pe + me, ind=ind, dir=n2o, ch4=ch4,
                         tot=ind + n2o + ch4, ef=100.0 * pr / tn_load,
                         kla3=ctrl[2], kla4=ctrl[3], kla5=ctrl[4], qintr=ctrl[5]))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    feas = [r for r in recs if r['snh'] <= 4.0 and r['snoe'] <= 18.0 and r['tss'] <= 30.0]
    print('%-6s %3d points in %6.1f s -> %s   feasible %d/%d'
          % (name, len(recs), time.time() - t0, out_csv, len(feas), len(recs)))
    if verbose:
        print('SO    SNO    SNH    SNOe   EQI     kWh    dir     tot    qintr')
        for r in recs:
            print('%.1f  %.2f  %6.2f %6.2f %7.1f %7.1f %7.1f %7.1f %8.0f' % (
                r['so'], r['sno'], r['snh'], r['snoe'], r['eqi'], r['kwh'], r['dir'], r['tot'],
                r['qintr']))
    if feas:
        for tag, b in (('min EQI', min(feas, key=lambda r: r['eqi'])),
                       ('min energy', min(feas, key=lambda r: r['kwh'])),
                       ('min total CO2', min(feas, key=lambda r: r['tot']))):
            print('  %-14s SO=%.2f SNO=%.2f | EQI=%.1f SNH=%.2f | kWh=%.1f | ind=%.1f dir=%.1f tot=%.1f EF=%.4f%%'
                  % (tag, b['so'], b['sno'], b['eqi'], b['snh'], b['kwh'], b['ind'], b['dir'],
                     b['tot'], b['ef']))
    return recs


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'fine'
    names = list(GRIDS) if which == 'all' else [which]
    for n in names:
        if n not in GRIDS:
            raise SystemExit('unknown grid %r; choose from %s or all' % (n, ', '.join(GRIDS)))
    y0 = np.load('_ss_A_n2o.npy')
    inf0, tn_load = reference_influent()
    for n in names:
        run_grid(n, y0, inf0, tn_load)
