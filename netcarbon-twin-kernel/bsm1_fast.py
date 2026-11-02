# -*- coding: utf-8 -*-
"""Numba-accelerated BSM1 plant RHS (same equations as bsm1_plant.py)."""
import numpy as np
from numba import njit

SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK, TSS, Q, TEMP = range(16)

MU_H, K_S, K_OH, K_NO, B_H = 4.0, 10.0, 0.2, 0.5, 0.3
MU_A, K_NH, K_OA, B_A = 0.5, 1.0, 0.4, 0.05
NY_G, K_A, K_H, K_X, NY_H = 0.8, 0.05, 3.0, 0.1, 0.8
Y_H, Y_A, F_P, I_XB, I_XP = 0.67, 0.24, 0.08, 0.08, 0.06
X_I2TSS = X_S2TSS = X_BH2TSS = X_BA2TSS = X_P2TSS = 0.75

SOSAT = 8.0
QIN0 = 18446.0
VOL1, VOL2, VOL3 = 1000.0, 1000.0, 1333.0
NOFLAYERS, FEEDLAYER = 10, 5
AREA, HEIGHT = 1500.0, 4.0
V0MAX, V0, RH, RP, FNS, XT, SBLIMIT = 250.0, 474.0, 0.000576, 0.00286, 0.00228, 3000.0, 3000.0
CARBON_CONC = 400000.0

# settler component -> ASM1 index for the 11 soluble components
SC = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11])
SA = np.array([SI, SS, SO, SNO, SNH, SND, SALK, TEMP, 16, 17, 18])


@njit(cache=True, fastmath=True)
def asm1_deriv_nb(y, y_in, kla, vol):
    yt = np.maximum(y, 0.0)
    xbh = max(yt[XBH], 1e-6)
    xs = max(yt[XS], 1e-6)
    p1 = MU_H * (yt[SS] / (K_S + yt[SS])) * (yt[SO] / (K_OH + yt[SO])) * yt[XBH]
    p2 = (MU_H * (yt[SS] / (K_S + yt[SS])) * (K_OH / (K_OH + yt[SO]))
          * (yt[SNO] / (K_NO + yt[SNO])) * NY_G * yt[XBH])
    p3 = MU_A * (yt[SNH] / (K_NH + yt[SNH])) * (yt[SO] / (K_OA + yt[SO])) * yt[XBA]
    p4 = B_H * yt[XBH]
    p5 = B_A * yt[XBA]
    p6 = K_A * yt[SND] * yt[XBH]
    xr = xs / xbh
    p7 = (K_H * (xr / (K_X + xr))
          * ((yt[SO] / (K_OH + yt[SO])) + NY_H * (K_OH / (K_OH + yt[SO])) * (yt[SNO] / (K_NO + yt[SNO])))
          * yt[XBH])
    p8 = p7 * (yt[XND] / xs)
    r = np.zeros(21)
    r[SS] = (-p1 - p2) / Y_H + p7
    r[XS] = (1.0 - F_P) * (p4 + p5) - p7
    r[XBH] = p1 + p2 - p4
    r[XBA] = p3 - p5
    r[XP] = F_P * (p4 + p5)
    r[SO] = -(1.0 - Y_H) / Y_H * p1 - (4.57 - Y_A) / Y_A * p3
    r[SNO] = -((1.0 - Y_H) / (2.86 * Y_H)) * p2 + p3 / Y_A
    r[SNH] = -I_XB * (p1 + p2) - (I_XB + 1.0 / Y_A) * p3 + p6
    r[SND] = -p6 + p8
    r[XND] = (I_XB - F_P * I_XP) * (p4 + p5) - p8
    r[SALK] = (-I_XB / 14.0 * p1
               + ((1.0 - Y_H) / (14.0 * 2.86 * Y_H) - I_XB / 14.0) * p2
               - (I_XB / 14.0 + 1.0 / (7.0 * Y_A)) * p3 + p6 / 14.0)
    dy = np.zeros(21)
    for j in range(7):
        dy[j] = (y_in[Q] * (y_in[j] - y[j])) / vol + r[j]
    dy[SO] = (y_in[Q] * (y_in[SO] - y[SO])) / vol + r[SO] + kla * (SOSAT - y[SO])
    for j in range(8, 13):
        dy[j] = (y_in[Q] * (y_in[j] - y[j])) / vol + r[j]
    return dy


@njit(cache=True, fastmath=True)
def settler_output_nb(ys, ys_in, q_r, q_w):
    n = NOFLAYERS
    ret = np.zeros(21)
    eff = np.zeros(21)
    for k in range(11):
        ret[SA[k]] = max(ys[SC[k] * n + n - 1], 0.0)
        eff[SA[k]] = max(ys[SC[k] * n], 0.0)
    ret[TSS] = max(ys[7 * n + n - 1], 0.0)
    eff[TSS] = max(ys[7 * n], 0.0)
    if ys_in[TSS] > 1e-9:
        for ai in (XI, XS, XBH, XBA, XP, XND, 19, 20):
            ret[ai] = ret[TSS] / ys_in[TSS] * ys_in[ai]
            eff[ai] = eff[TSS] / ys_in[TSS] * ys_in[ai]
    ret[Q] = q_r
    eff[Q] = ys_in[Q] - q_w - q_r
    return ret, eff


@njit(cache=True, fastmath=True)
def settler_deriv_nb(ys, ys_in, q_r, q_w):
    n = NOFLAYERS
    h = HEIGHT / n
    ys = np.maximum(ys, 0.0)
    v_in = ys_in[Q] / AREA
    q_u = q_r + q_w
    v_up = (ys_in[Q] - q_u) / AREA
    v_dn = q_u / AREA
    tss = ys[7 * n:8 * n]
    arg = tss - FNS * ys_in[TSS]
    vs = V0 * (np.exp(-RH * arg) - np.exp(-RP * arg))
    js_t = np.zeros(n)
    for i in range(n):
        v = vs[i]
        if v < 0.0:
            v = 0.0
        elif v > V0MAX:
            v = V0MAX
        js_t[i] = v * tss[i]
    js = np.zeros(n + 1)
    for i in range(n - 1):
        if i < FEEDLAYER - 1 and tss[i + 1] <= XT:
            js[i + 1] = js_t[i]
        else:
            js[i + 1] = min(js_t[i], js_t[i + 1])
    dys = np.zeros(12 * n)
    for k in range(11):
        b = SC[k] * n
        for i in range(FEEDLAYER - 1):
            dys[b + i] = (-v_up * ys[b + i] + v_up * ys[b + i + 1]) / h
        f = FEEDLAYER - 1
        dys[b + f] = (v_in * ys_in[SA[k]] - v_up * ys[b + f] - v_dn * ys[b + f]) / h
        for i in range(FEEDLAYER, n):
            dys[b + i] = (v_dn * ys[b + i - 1] - v_dn * ys[b + i]) / h
    b = 7 * n
    for i in range(FEEDLAYER - 1):
        dys[b + i] = (-v_up * ys[b + i] + v_up * ys[b + i + 1] - js[i + 1] + js[i]) / h
    f = FEEDLAYER - 1
    dys[b + f] = (v_in * ys_in[TSS] - v_up * ys[b + f] - v_dn * ys[b + f] - js[f + 1] + js[f]) / h
    for i in range(FEEDLAYER, n):
        dys[b + i] = (v_dn * ys[b + i - 1] - v_dn * ys[b + i] - js[i + 1] + js[i]) / h
    return dys


@njit(cache=True, fastmath=True)
def plant_deriv_nb(y, t, inf, u):
    kla = u[0:5]
    q_intr, q_r, q_w, carb1 = u[5], u[6], u[7], u[8]
    q_in = inf[Q]
    yr = y[:105].reshape(5, 21)
    for i in range(5):
        yr[i][TSS] = (X_I2TSS * yr[i][XI] + X_S2TSS * yr[i][XS] + X_BH2TSS * yr[i][XBH]
                      + X_BA2TSS * yr[i][XBA] + X_P2TSS * yr[i][XP])
    ys = y[105:]
    y_feed = yr[4].copy()
    y_feed[Q] = q_in + q_r
    q1 = q_in + q_r + q_intr
    y_in1 = np.zeros(21)
    for j in range(21):
        y_in1[j] = (inf[j] * q_in + yr[4][j] * q_intr) / q1
    ret, _eff = settler_output_nb(ys, y_feed, q_r, q_w)
    for j in range(21):
        y_in1[j] = y_in1[j] + ret[j] * q_r / q1
    y_in1[Q] = q1
    if carb1 > 0.0:
        qt = q1 + carb1
        y_in1[SI] = y_in1[SI] * q1 / qt
        y_in1[SS] = (y_in1[SS] * q1 + CARBON_CONC * carb1) / qt
        for j in range(3, 14):
            y_in1[j] = y_in1[j] * q1 / qt
        y_in1[Q] = qt
    dyr = np.zeros((5, 21))
    yin = y_in1
    vols = (VOL1, VOL2, VOL3, VOL3, VOL3)
    for i in range(5):
        dyr[i] = asm1_deriv_nb(yr[i], yin, kla[i], vols[i])
        yin = yr[i]
    dys = settler_deriv_nb(ys, y_feed, q_r, q_w)
    out = np.zeros(225)
    for i in range(5):
        for j in range(21):
            out[i * 21 + j] = dyr[i, j]
    for j in range(120):
        out[105 + j] = dys[j]
    return out


@njit(cache=True, fastmath=True)
def tss_vec(y):
    yr = y[:105].reshape(5, 21)
    for i in range(5):
        yr[i][TSS] = (X_I2TSS * yr[i][XI] + X_S2TSS * yr[i][XS] + X_BH2TSS * yr[i][XBH]
                      + X_BA2TSS * yr[i][XBA] + X_P2TSS * yr[i][XP])
    return y


@njit(cache=True, fastmath=True)
def rk4_step(y, dt, inf, u, nsub):
    """Fixed-step RK4 integration of the plant over dt (days) with nsub substeps."""
    h = dt / nsub
    for _ in range(nsub):
        k1 = plant_deriv_nb(y, 0.0, inf, u)
        k2 = plant_deriv_nb(y + 0.5 * h * k1, 0.0, inf, u)
        k3 = plant_deriv_nb(y + 0.5 * h * k2, 0.0, inf, u)
        k4 = plant_deriv_nb(y + h * k3, 0.0, inf, u)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for j in range(225):
            if y[j] < 0.0:
                y[j] = 0.0
    return y


@njit(cache=True, fastmath=True)
def settle_metrics(y, inf, u):
    """Instantaneous plant outputs: (effluent vector, recycle, TSS profile, energy kWh/d, EQI)."""
    yr = y[:105].reshape(5, 21)
    for i in range(5):
        yr[i][TSS] = (X_I2TSS * yr[i][XI] + X_S2TSS * yr[i][XS] + X_BH2TSS * yr[i][XBH]
                      + X_BA2TSS * yr[i][XBA] + X_P2TSS * yr[i][XP])
    ys = y[105:]
    y_feed = yr[4].copy()
    y_feed[Q] = inf[Q] + u[6]
    ret, eff = settler_output_nb(ys, y_feed, u[6], u[7])
    ae = (SOSAT * (VOL1 * u[0] + VOL2 * u[1] + VOL3 * (u[2] + u[3] + u[4]))) / (1.8 * 1000.0)
    pe = u[5] * 0.004 + u[6] * 0.008 + u[7] * 0.05
    n_unaer = 0
    if u[0] < 20.0:
        n_unaer += 1
    if u[1] < 20.0:
        n_unaer += 1
    me = 0.005 * (VOL1 * (1 if u[0] < 20.0 else 0) + VOL2 * (1 if u[1] < 20.0 else 0)
                  + VOL3 * (1 if u[2] < 20.0 else 0) + VOL3 * (1 if u[3] < 20.0 else 0)
                  + VOL3 * (1 if u[4] < 20.0 else 0))
    return eff, ret, me * 24.0, ae, pe


@njit(cache=True, fastmath=True)
def eqi_nb(eff):
    nkj = (eff[SNH] + eff[SND] + eff[XND] + I_XB * (eff[XBH] + eff[XBA]) + I_XP * (eff[XP] + eff[XI]))
    cod = eff[SS] + eff[SI] + eff[XS] + eff[XI] + eff[XBH] + eff[XBA] + eff[XP]
    bod5 = 0.65 * (eff[SS] + eff[XS] + (1.0 - F_P) * (eff[XBH] + eff[XBA]))
    return (2.0 * eff[TSS] + cod + 30.0 * nkj + 10.0 * eff[SNO] + 2.0 * bod5) * eff[Q] / 1000.0
