# -*- coding: utf-8 -*-
"""Faithful replication of the published BSM1 closed loop (bsm2_python BSM1CL).

Arm A = BSM1 default control: KLa3/KLa4 fixed at KLA3_INIT, KLa5 <- PI on SO5,
QINTR <- PI on SNO2, with the official sensor/actuator transfer functions.

Flat-array layout (built by make_disc_flat):
  ad: [SNO2 8x8=64][SO5 2x2=4][KLA5 2x2=4][QINTR 1x1=1]        total 73
  bd: [SNO2 8][SO5 2][KLA5 2][QINTR 1]                          total 13
  cd: [SNO2 8][SO5 2][KLA5 2][QINTR 1]                          total 13
  dd: [SNO2][SO5][KLA5][QINTR]                                  total 4
"""
import numpy as np
from scipy.signal import tf2ss, cont2discrete
from numba import njit
from bsm1_fast import SO, SNO, QIN0, rk4_step

QW = 385.0
QR = QIN0
DT = 1.0 / 1440.0

# --- official gains -----------------------------------------------------------
KSO, TISO, TTSO, SO5REF = 25.0, 0.002, 0.001, 2.0
KLA_MIN, KLA_MAX, KLA_OFFSET = 0.0, 360.0, 144.0
KLA_INIT = 201.3015557168598

KQ, TIQ, TTQ, SNO2REF = 10000.0, 0.025, 0.015, 1.0
QINTR_MIN, QINTR_MAX, QINTR_OFFSET = 0.0, 5.0 * QIN0, 18500.0

# --- official sensor/actuator time constants [d] ------------------------------
T_SNO2 = 10.0 / (60.0 * 24.0) / 11.7724
T_SO5 = 1.0 / (60.0 * 24.0) / 3.89
T_KLA5 = 4.0 / (60.0 * 24.0) / 3.89
T_QINTR = 0.001

MIN_SNO2, MAX_SNO2 = 0.0, 10.0
MIN_SO5, MAX_SO5 = 0.0, 10.0

N_S, N_O, N_K, N_Q = 8, 2, 2, 1

# --- ad / bd offsets (block sizes n*n and n) ----------------------------------
A_AS, A_AO, A_AK, A_AQ = 0, 64, 68, 72
B_BS, B_BO, B_BK, B_BQ = 0, 8, 10, 12
# --- cd / dd offsets (block size n each) --------------------------------------
C_CS, C_CO, C_CK, C_CQ = 0, 8, 10, 12
D_DS, D_DO, D_DK, D_DQ = 0, 1, 2, 3


def make_den():
    d_sno2 = [T_SNO2 ** 8, 8 * T_SNO2 ** 7, 28 * T_SNO2 ** 6, 56 * T_SNO2 ** 5,
              70 * T_SNO2 ** 4, 56 * T_SNO2 ** 3, 28 * T_SNO2 ** 2, 8 * T_SNO2, 1.0]
    return d_sno2, [T_SO5 ** 2, 2 * T_SO5, 1.0], [T_KLA5 ** 2, 2 * T_KLA5, 1.0], [T_QINTR, 1.0]


def _zoh(num, den, dt):
    A, B, C, D = tf2ss(num, den)
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method='zoh')
    return (np.ascontiguousarray(Ad, dtype=np.float64),
            np.ascontiguousarray(Bd.ravel(), dtype=np.float64),
            np.ascontiguousarray(Cd.ravel(), dtype=np.float64),
            float(np.asarray(Dd).ravel()[0]))


def make_disc_flat(dt):
    d_sno2, d_so5, d_kla5, d_qintr = make_den()
    blks = [_zoh(1.0, d_sno2, dt), _zoh(1.0, d_so5, dt), _zoh(1.0, d_kla5, dt), _zoh(1.0, d_qintr, dt)]
    ad = np.concatenate([b[0].ravel() for b in blks])
    bd = np.concatenate([b[1].ravel() for b in blks])
    cd = np.concatenate([b[2].ravel() for b in blks])
    dd = np.array([b[3] for b in blks])
    assert ad.size == 73 and bd.size == 13 and cd.size == 13 and dd.size == 4, (ad.size, bd.size, cd.size, dd.size)
    return ad, bd, cd, dd


@njit(cache=True, fastmath=True)
def _blkstep(x, Ad, Bd, Cd, Dd, u):
    """One ZOH step of a SISO discrete state-space block (contiguous slices)."""
    n = x.size
    xn = np.empty(n)
    y = 0.0
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += Ad[i * n + j] * x[j]
        xn[i] = acc + Bd[i] * u
        y += Cd[i] * xn[i]
    y += Dd * u
    return xn, y


@njit(cache=True, fastmath=True)
def _pid(k, t_i, t_d, t_t, offset, lo, hi, setpoint, signal, dt, inject,
         prev_error, prev_signal, prev_lim):
    error = k * (setpoint - signal)
    integral = error / t_i * dt
    derivative = k * t_d * (error - prev_error) / dt
    aw = (prev_lim - prev_signal) / t_t * dt
    cs = error + integral + derivative + offset + aw + inject
    ps = cs
    if cs < lo:
        cs = lo
    elif cs > hi:
        cs = hi
    return cs, error, ps, cs


@njit(cache=True, fastmath=True)
def arm_A(y0, inf, days, dt, nsub, ad, bd, cd, dd, kla34, x_s0, x_o0, x_k0, x_q0):
    y = y0.copy()
    kla5 = KLA_INIT
    qintr = 3.0 * QIN0

    xs = x_s0.copy()
    xo = x_o0.copy()
    xk = x_k0.copy()
    xq = x_q0.copy()

    e_so, ps_so, pl_so = 0.0, 0.0, 0.0
    e_q, ps_q, pl_q = 0.0, 0.0, 0.0

    u = np.zeros(9)
    u[2] = kla34
    u[3] = kla34
    u[6] = QR
    u[7] = QW

    n = int(days / dt)
    for _ in range(n):
        yr = y[:105].reshape(5, 21)

        xs, s = _blkstep(xs, ad[A_AS:A_AS + 64], bd[B_BS:B_BS + 8], cd[C_CS:C_CS + 8], dd[D_DS], yr[1][SNO])
        if s < MIN_SNO2:
            s = MIN_SNO2
        elif s > MAX_SNO2:
            s = MAX_SNO2
        cs, e_q, ps_q, pl_q = _pid(KQ, TIQ, 0.0, TTQ, QINTR_OFFSET, QINTR_MIN, QINTR_MAX,
                                   SNO2REF, s, dt, 0.0, e_q, ps_q, pl_q)
        xq, qintr = _blkstep(xq, ad[A_AQ:A_AQ + 1], bd[B_BQ:B_BQ + 1], cd[C_CQ:C_CQ + 1], dd[D_DQ], cs)

        xo, s5 = _blkstep(xo, ad[A_AO:A_AO + 4], bd[B_BO:B_BO + 2], cd[C_CO:C_CO + 2], dd[D_DO], yr[4][SO])
        if s5 < MIN_SO5:
            s5 = MIN_SO5
        elif s5 > MAX_SO5:
            s5 = MAX_SO5
        cs2, e_so, ps_so, pl_so = _pid(KSO, TISO, 0.0, TTSO, KLA_OFFSET, KLA_MIN, KLA_MAX,
                                       SO5REF, s5, dt, 0.0, e_so, ps_so, pl_so)
        xk, kla5 = _blkstep(xk, ad[A_AK:A_AK + 4], bd[B_BK:B_BK + 2], cd[C_CK:C_CK + 2], dd[D_DK], cs2)

        u[4] = kla5
        u[5] = qintr
        y = rk4_step(y, dt, inf, u, nsub)

    ctrl = np.array([0.0, 0.0, kla34, kla34, kla5, qintr, QR, QW, 0.0])
    return y, ctrl, xs, xo, xk, xq
