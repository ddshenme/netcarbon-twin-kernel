# -*- coding: utf-8 -*-
"""Hierarchical closed-loop runner for the BSM1 + N2O-overlay plant.

Architecture (shared by every arm; only the upper level differs)
----------------------------------------------------------------
  * low level, at the plant time step (1 min): the official BSM1 PI loops
    (aerationcontrolinit_bsm1 gains, official sensor/actuator transfer functions,
    anti-windup) hold dissolved oxygen in the aerobic reactors and nitrate in reactor 2
    at whatever setpoints the upper level asks for;
  * upper level: constant setpoints (baseline / sweeps) or a learned policy (arms B/C/D),
    updated once per decision interval.

Because the low level is identical for all arms, the comparison is apples-to-apples: the
arms differ only in the setpoint policy imposed on the same physical control layer.

Two entry points, one control layer
-----------------------------------
  cl_epoch(y0, inf, days, ...)                  constant influent, returns the final state.
                                                Used by the steady-state operating-map sweeps
                                                and by reference/_run_armA.py.
  cl_steps(y0, inf_arr, ef_arr, nstep, ...)     minute-by-minute influent and grid factor,
                                                integrates every objective term into 'acc'.
                                                Used by env.py for the dynamic campaigns.

nmode = 0  -> reactors 3 and 4 held at kla34 (official BSM1CL behaviour, arm A);
              only reactor 5 follows a PI loop.
nmode = 3  -> reactors 3, 4 and 5 each follow their own PI loop (arms B/C/D).

acc layout (cl_steps), 16 entries
---------------------------------
  0 aeration energy          [kWh]
  1 pumping energy           [kWh]
  2 mixing energy            [kWh]
  3 indirect CO2             [kgCO2]     (energy x time-varying grid factor)
  4 direct N2O-N produced    [gN]
  5 direct CH4               [gN-equivalent]
  6 effluent volume          [m3]
  7 EQI                      [kg]
  8 effluent NH4-N integral  [gN/m3 * d]
  9 effluent NO3-N integral  [gN/m3 * d]
 10 effluent TSS integral    [g/m3 * d]
 11 number of steps
 12 time with effluent NH4-N above 4 gN/m3      [d]
 13 time with effluent total N above 18 gN/m3   [d]
 14 time with effluent TSS above 30 g/m3        [d]
 15 effluent total-N integral                   [gN/m3 * d]
"""
import numpy as np
from numba import njit
from bsm1_fast import (SO, SNO, SNH, TSS, Q, SND, XND, XBH, XBA, XP, XI, QIN0,
                       VOL1, VOL2, VOL3, SOSAT, settle_metrics, eqi_nb)
from control_arms import (make_disc_flat,
                          A_AS, A_AO, A_AK, A_AQ, B_BS, B_BO, B_BK, B_BQ,
                          C_CS, C_CO, C_CK, C_CQ, D_DS, D_DO, D_DK, D_DQ,
                          KSO, TISO, TTSO, KQ, TIQ, TTQ,
                          KLA_MIN, KLA_MAX, KLA_OFFSET, KLA_INIT,
                          QINTR_MIN, QINTR_MAX, QINTR_OFFSET,
                          MIN_SNO2, MAX_SNO2, MIN_SO5, MAX_SO5, QW, QR, DT)
import n2o as N

EF_GRID = 0.5306
SP_SO3, SP_SO4, SP_SO5, SP_SNO2 = 0, 1, 2, 3
NAER = 3          # aerobic PI loops: reactors 3, 4, 5
NXA = 12          # aeration controller states: (SO sensor 2 + KLA actuator 2) x 3
# per-loop offsets inside xa
XA_S = (0, 4, 8)      # sensor state offset
XA_K = (2, 6, 10)     # actuator state offset

NACC = 16             # length of the cl_steps accumulator vector
LIM_SNH, LIM_TN, LIM_TSS = 4.0, 18.0, 30.0   # BSM1 effluent limits [g/m3]


@njit(cache=True, fastmath=True)
def _pid(k, t_i, t_t, offset, lo, hi, setpoint, signal, dt, prev_error, prev_signal, prev_lim):
    """One step of the official PID: the integral term is recomputed, not accumulated."""
    error = k * (setpoint - signal)
    integral = error / t_i * dt
    aw = (prev_lim - prev_signal) / t_t * dt
    cs = error + integral + offset + aw
    ps = cs
    if cs < lo:
        cs = lo
    elif cs > hi:
        cs = hi
    return cs, error, ps, cs


@njit(cache=True, fastmath=True)
def _siso(x, xo, n, Ad, aoff, Bd, boff, Cd, coff, Dd, doff, u):
    """One ZOH step of a SISO block whose state lives in x[xo:xo+n]; returns the output."""
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += Ad[aoff + i * n + j] * x[xo + j]
        x[xo + i] = acc + Bd[boff + i] * u
    y = 0.0
    for i in range(n):
        y += Cd[coff + i] * x[xo + i]
    y += Dd[doff] * u
    return y


@njit(cache=True, fastmath=True)
def _aerl(yr, xa, ma, sp, dt, ad, bd, cd, dd, on3, on4, kla34):
    """One step of the three aeration PI loops; returns (kla3, kla4, kla5)."""
    kla = np.zeros(3)
    for i in range(NAER):
        if i == 0 and not on3:
            kla[0] = kla34
        elif i == 1 and not on4:
            kla[1] = kla34
        else:
            s = _siso(xa, XA_S[i], 2, ad, A_AO, bd, B_BO, cd, C_CO, dd, D_DO, yr[i + 2][SO])
            if s < MIN_SO5:
                s = MIN_SO5
            elif s > MAX_SO5:
                s = MAX_SO5
            cs, ma[i, 0], ma[i, 1], ma[i, 2] = _pid(KSO, TISO, TTSO, KLA_OFFSET, KLA_MIN,
                                                    KLA_MAX, sp[SP_SO3 + i], s, dt,
                                                    ma[i, 0], ma[i, 1], ma[i, 2])
            kla[i] = _siso(xa, XA_K[i], 2, ad, A_AK, bd, B_BK, cd, C_CK, dd, D_DK, cs)
    return kla[0], kla[1], kla[2]


@njit(cache=True, fastmath=True)
def cl_epoch(y0, inf, days, dt, nsub, ad, bd, cd, dd, sp, nmode, kla34,
             xs, xa, xq, ma, mq):
    """Advance the closed loop by 'days' with constant setpoints sp = [SO3, SO4, SO5, SNO2]."""
    y = y0.copy()
    n = int(days / dt)
    on3 = nmode >= 3
    on4 = nmode >= 3
    kla3 = kla34
    kla4 = kla34
    kla5 = kla34
    qintr = QIN0
    for _ in range(n):
        yr = y[:105].reshape(5, 21)
        # --- nitrate / internal-recycle loop (reactor 2) ---
        s = _siso(xs, 0, 8, ad, A_AS, bd, B_BS, cd, C_CS, dd, D_DS, yr[1][SNO])
        if s < MIN_SNO2:
            s = MIN_SNO2
        elif s > MAX_SNO2:
            s = MAX_SNO2
        cs, mq[0], mq[1], mq[2] = _pid(KQ, TIQ, TTQ, QINTR_OFFSET, QINTR_MIN, QINTR_MAX,
                                       sp[SP_SNO2], s, dt, mq[0], mq[1], mq[2])
        qintr = _siso(xq, 0, 1, ad, A_AQ, bd, B_BQ, cd, C_CQ, dd, D_DQ, cs)
        # --- aeration loops ---
        kla3, kla4, kla5 = _aerl(yr, xa, ma, sp, dt, ad, bd, cd, dd, on3, on4, kla34)
        u = np.zeros(9)
        u[2] = kla3
        u[3] = kla4
        u[4] = kla5
        u[5] = qintr
        u[6] = QR
        u[7] = QW
        y = N.rk4_step_ext(y, dt, inf, u, nsub)
    ctrl = np.array([0.0, 0.0, kla3, kla4, kla5, qintr, QR, QW, 0.0])
    return y, ctrl, xs, xa, xq, ma, mq


@njit(cache=True, fastmath=True)
def cl_steps(y0, inf_arr, ef_arr, nstep, dt, nsub, ad, bd, cd, dd, sp, nmode, kla34,
             xs, xa, xq, ma, mq, acc):
    """Run 'nstep' 1-minute steps and accumulate the objective terms into acc."""
    y = y0.copy()
    on3 = nmode >= 3
    on4 = nmode >= 3
    kla3 = kla34
    kla4 = kla34
    kla5 = kla34
    qintr = 3.0 * 18446.0
    for i in range(nstep):
        inf = inf_arr[i]
        yr = y[:105].reshape(5, 21)
        s = _siso(xs, 0, 8, ad, A_AS, bd, B_BS, cd, C_CS, dd, D_DS, yr[1][SNO])
        if s < MIN_SNO2:
            s = MIN_SNO2
        elif s > MAX_SNO2:
            s = MAX_SNO2
        cs, mq[0], mq[1], mq[2] = _pid(KQ, TIQ, TTQ, QINTR_OFFSET, QINTR_MIN, QINTR_MAX,
                                       sp[SP_SNO2], s, dt, mq[0], mq[1], mq[2])
        qintr = _siso(xq, 0, 1, ad, A_AQ, bd, B_BQ, cd, C_CQ, dd, D_DQ, cs)
        kla3, kla4, kla5 = _aerl(yr, xa, ma, sp, dt, ad, bd, cd, dd, on3, on4, kla34)
        u = np.zeros(9)
        u[2] = kla3
        u[3] = kla4
        u[4] = kla5
        u[5] = qintr
        u[6] = QR
        u[7] = QW
        y = N.rk4_step_ext(y, dt, inf, u, nsub)

        eff, ret, me, ae, pe = settle_metrics(y[:225], inf, u)
        st, ef_n2o, wa, pr, a, b, c = N.n2o_fluxes(y, inf, u)
        acc[0] += ae * dt
        acc[1] += pe * dt
        acc[2] += me * dt
        acc[3] += (ae + pe + me) * ef_arr[i] * dt
        acc[4] += pr * dt
        acc[5] += N.ch4_flux(y, inf, u) * dt
        acc[6] += eff[Q] * dt
        acc[7] += eqi_nb(eff) * dt
        acc[8] += eff[SNH] * dt
        acc[9] += eff[SNO] * dt
        acc[10] += eff[TSS] * dt
        nkj = (eff[SNH] + eff[SND] + eff[XND] + 0.08 * (eff[XBH] + eff[XBA])
               + 0.06 * (eff[XP] + eff[XI]))
        if eff[SNH] > LIM_SNH:
            acc[12] += dt
        if nkj + eff[SNO] > LIM_TN:
            acc[13] += dt
        acc[15] += (nkj + eff[SNO]) * dt
        if eff[TSS] > LIM_TSS:
            acc[14] += dt
    acc[11] += nstep
    ctrl = np.array([0.0, 0.0, kla3, kla4, kla5, qintr, QR, QW, 0.0])
    return y, ctrl, xs, xa, xq, ma, mq, acc


@njit(cache=True, fastmath=True)
def metrics(y, inf, u):
    """(effluent, recycle, ME, AE, PE) in kWh/d for a 255-state extended plant state."""
    return settle_metrics(y[:225], inf, u)


@njit(cache=True, fastmath=True)
def kwh(y, inf, u):
    eff, ret, me, ae, pe = settle_metrics(y[:225], inf, u)
    return ae + pe + me


@njit(cache=True, fastmath=True)
def carbon(y, inf, u):
    """(indirect kgCO2/d, direct N2O kgCO2e/d, CH4 kgCO2e/d, N2O-N produced gN/d)."""
    eff, ret, me, ae, pe = settle_metrics(y[:225], inf, u)
    ind = (ae + pe + me) * EF_GRID
    st, ef, wa, pr, a, b, c = N.n2o_fluxes(y, inf, u)
    n2o_co2 = (st + ef + wa) / 1000.0 * (44.0 / 28.0) * N.GWP_N2O
    ch4_co2 = N.ch4_flux(y, inf, u) / 1000.0 * (16.0 / 14.0) * N.GWP_CH4
    return ind, n2o_co2, ch4_co2, pr


def fresh_state():
    """Zero-initialised controller memory (the official sensors/actuators start at zero).

    Returns the controller state both entry points need, in cl_steps argument order:
    (ad, bd, cd, dd, xs, xa, xq, ma, mq). cl_epoch takes the same tuple minus the leading
    four matrices, i.e. *(state[4:]).
    """
    ad, bd, cd, dd = make_disc_flat(DT)
    return ad, bd, cd, dd, np.zeros(8), np.zeros(NXA), np.zeros(1), np.zeros((NAER, 3)), np.zeros(3)
