# -*- coding: utf-8 -*-
"""
BSM1 benchmark plant: 5-reactor ASM1 + 10-layer Takacs secondary clarifier
+ aeration/pumping/mixing energy + indirect & direct GHG accounting.

Model equations are a pure-NumPy port of the IWA/COST BSM1 reference
(as implemented in the FAU bsm2-python reference code, which is itself a port
of the Lund University BSM1/BSM2 code).  All stoichiometric/kinetic constants
are the BSM1 values at 15 degC (Alex et al. 2018).
"""
import numpy as np
from scipy.integrate import odeint

# ------------------------------------------------------------------ ASM1 indices
SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK, TSS, Q, TEMP, SD1, SD2, SD3, XD4, XD5 = range(21)

# --------------------------------------------------- BSM1 parameters at 15 degC
MU_H, K_S, K_OH, K_NO, B_H = 4.0, 10.0, 0.2, 0.5, 0.3
MU_A, K_NH, K_OA, B_A = 0.5, 1.0, 0.4, 0.05
NY_G, K_A, K_H, K_X, NY_H = 0.8, 0.05, 3.0, 0.1, 0.8
Y_H, Y_A, F_P, I_XB, I_XP = 0.67, 0.24, 0.08, 0.08, 0.06
X_I2TSS = X_S2TSS = X_BH2TSS = X_BA2TSS = X_P2TSS = 0.75
PAR = np.array([MU_H, K_S, K_OH, K_NO, B_H, MU_A, K_NH, K_OA, B_A, NY_G, K_A, K_H,
                K_X, NY_H, Y_H, Y_A, F_P, I_XB, I_XP,
                X_I2TSS, X_S2TSS, X_BH2TSS, X_BA2TSS, X_P2TSS])

# ------------------------------------------------------------ plant configuration
VOL = np.array([1000.0, 1000.0, 1333.0, 1333.0, 1333.0])   # m3
SOSAT = 8.0                                                # g O2/m3 (15 degC)
QIN0 = 18446.0
QR_DEF = 18446.0
QW_DEF = 385.0
QINTR_DEF = 3.0 * QIN0
KLA_DEF = np.array([0.0, 0.0, 240.0, 240.0, 84.0])
CARBON_CONC = 400000.0                                     # g COD/m3
QINTR_MAX = 5.0 * QIN0
QR_MAX = 2.0 * QIN0
QW_MAX = 0.1 * QIN0

# ----------------------------------------------- secondary clarifier (Takacs 1D)
AREA, HEIGHT = 1500.0, 4.0
NOFLAYERS, FEEDLAYER = 10, 5
V0MAX, V0, RH, RP, FNS, XT, SBLIMIT = 250.0, 474.0, 0.000576, 0.00286, 0.00228, 3000.0, 3000.0
SEDPAR = np.array([V0MAX, V0, RH, RP, FNS, XT, SBLIMIT])

# settler component order -> ASM1 index (component 7 is TSS, handled separately)
SETTLER_SOLUBLE = [(0, SI), (1, SS), (2, SO), (3, SNO), (4, SNH), (5, SND),
                   (6, SALK), (8, TEMP), (9, SD1), (10, SD2), (11, SD3)]

# --------------------------------------------------------- energy / GHG factors
PF_QINTR, PF_QR, PF_QW = 0.004, 0.008, 0.05     # kWh/m3
ME_FACTOR = 0.005                                # kWh/m3 (non-aerated tanks)
AE_SPEC = 1.8                                    # kg O2 / kWh
EF_GRID = 0.5306                                 # kg CO2/kWh, MEE+NBS Announcement 2025 No.47
GWP100_N2O = 273.0                               # IPCC AR6
MW_N2O_N = 44.0 / 28.0

# effluent limits (BSM1)
LIM_TOTALN, LIM_SNH, LIM_TSS, LIM_COD, LIM_BOD5 = 18.0, 4.0, 30.0, 100.0, 10.0


# =============================================================================
#  ASM1 reactor
# =============================================================================
def asm1_deriv(y, y_in, kla, vol):
    """Right-hand side of the ASM1 mass balances for one completely mixed reactor."""
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
    p7 = (K_H * ((xs / xbh) / (K_X + (xs / xbh)))
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
    dy[0:7] = (y_in[Q] * (y_in[0:7] - y[0:7])) / vol + r[0:7]
    dy[SO] = (y_in[Q] * (y_in[SO] - y[SO])) / vol + r[SO] + kla * (SOSAT - y[SO])
    dy[8:13] = (y_in[Q] * (y_in[8:13] - y[8:13])) / vol + r[8:13]
    return dy


def tss_from_particulates(y):
    return X_I2TSS * y[XI] + X_S2TSS * y[XS] + X_BH2TSS * y[XBH] + X_BA2TSS * y[XBA] + X_P2TSS * y[XP]


def add_carbon(y_in, carb, conc=CARBON_CONC):
    """Mix an external carbon source into a stream (concentrations weighted by flow)."""
    y = y_in.copy()
    qt = y[Q] + carb
    y[SI] = y[SI] * y_in[Q] / qt
    y[SS] = (y[SS] * y_in[Q] + conc * carb) / qt
    y[3:14] = y[3:14] * y_in[Q] / qt
    y[Q] = qt
    return y


# =============================================================================
#  Secondary clarifier
# =============================================================================
def settler_initial_state():
    ys = np.zeros(12 * NOFLAYERS)
    n = NOFLAYERS
    ys[0 * n:1 * n] = 28.0643
    ys[1 * n:2 * n] = 0.6734
    ys[2 * n:3 * n] = 1.3748
    ys[3 * n:4 * n] = 9.1948
    ys[4 * n:5 * n] = 0.1585
    ys[5 * n:6 * n] = 0.5594
    ys[6 * n:7 * n] = 4.5646
    ys[7 * n:8 * n] = [14.3255, 20.8756, 34.2948, 81.0276, 423.2035, 423.2035,
                       423.2035, 423.2035, 3710.6, 7348.3]
    ys[8 * n:9 * n] = 15.0
    return ys


def settler_output(ys, ys_in, q_r, q_w):
    """Return (recycle, waste, effluent, TSS profile, sludge blanket level)."""
    ys = np.maximum(ys, 0.0)
    n = NOFLAYERS
    ret, was, eff = np.zeros(21), np.zeros(21), np.zeros(21)
    for c, ai in SETTLER_SOLUBLE:
        ret[ai] = ys[c * n + n - 1]
        eff[ai] = ys[c * n + 0]
    ret[TSS] = ys[7 * n + n - 1]
    eff[TSS] = ys[7 * n + 0]
    if ys_in[TSS] > 1e-9:
        for ai in (XI, XS, XBH, XBA, XP, XND, XD4, XD5):
            ret[ai] = ret[TSS] / ys_in[TSS] * ys_in[ai]
            eff[ai] = eff[TSS] / ys_in[TSS] * ys_in[ai]
    ret[Q] = q_r
    was[:] = ret
    was[Q] = q_w
    eff[Q] = ys_in[Q] - q_w - q_r
    tss_profile = ys[7 * n:8 * n].copy()
    below = np.where(tss_profile < SBLIMIT)[0]
    level = (n - 1 - int(np.max(below))) if len(below) else n
    return ret, was, eff, tss_profile, level


def settler_deriv(ys, ys_in, q_r, q_w):
    ys = np.maximum(ys, 0.0)
    n = NOFLAYERS
    h = HEIGHT / n
    v_in = ys_in[Q] / AREA
    q_u = q_r + q_w
    q_e = ys_in[Q] - q_u
    v_up, v_dn = q_e / AREA, q_u / AREA
    tss = ys[7 * n:8 * n]
    arg = tss - FNS * ys_in[TSS]
    vs = V0 * (np.exp(-RH * arg) - np.exp(-RP * arg))
    vs = np.clip(vs, 0.0, V0MAX)
    js_t = vs * tss
    js = np.zeros(n + 1)
    for i in range(n - 1):
        if i < FEEDLAYER - 1 - 1e-2 and tss[i + 1] <= XT:
            js[i + 1] = js_t[i]
        else:
            js[i + 1] = min(js_t[i], js_t[i + 1])
    dys = np.zeros(12 * n)
    for c, ai in SETTLER_SOLUBLE:
        b = c * n
        for i in range(FEEDLAYER - 1):
            dys[b + i] = (-v_up * ys[b + i] + v_up * ys[b + i + 1]) / h
        f = FEEDLAYER - 1
        dys[b + f] = (v_in * ys_in[ai] - v_up * ys[b + f] - v_dn * ys[b + f]) / h
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


# =============================================================================
#  Full plant
# =============================================================================
def plant_deriv(y, t, inf, u):
    """State = 5*21 ASM1 reactor states followed by 12*10 clarifier states."""
    kla = u[0:5]
    q_intr, q_r, q_w, carb1 = u[5], u[6], u[7], u[8]
    q_in = inf[Q]
    y = np.asarray(y, dtype=float).ravel()
    yr = y[:105].reshape(5, 21)
    for i in range(5):
        yr[i][TSS] = tss_from_particulates(yr[i])
    ys = y[105:]
    y_feed = yr[4].copy()
    y_feed[Q] = q_in + q_r
    q1 = q_in + q_r + q_intr
    y_in1 = (inf * q_in + yr[4] * q_intr) / q1
    ret, _, _, _, _ = settler_output(ys, y_feed, q_r, q_w)
    y_in1 = y_in1 + ret * q_r / q1
    y_in1[Q] = q1
    if carb1 > 0.0:
        y_in1 = add_carbon(y_in1, carb1)
    dyr = np.zeros((5, 21))
    yin = y_in1
    for i in range(5):
        dyr[i] = asm1_deriv(yr[i], yin, kla[i], VOL[i])
        yin = yr[i]
    dys = settler_deriv(ys, y_feed, q_r, q_w)
    return np.concatenate([dyr.ravel(), dys])


def plant_step(y, dt, inf, u, rtol=1e-4, atol=1e-4):
    sol = odeint(plant_deriv, y, [0.0, dt], args=(inf, u), rtol=rtol, atol=atol, mxstep=4000)
    y = np.maximum(sol[-1], 0.0)
    yr = y[:105].reshape(5, 21)
    for i in range(5):
        yr[i][TSS] = tss_from_particulates(yr[i])
    y[:105] = yr.ravel()
    return y


# ------------------------------------------------------------------ energy & GHG
def aeration_energy(kla):
    """kWh/d: sum over aerated reactors of S_O,sat * V * KLa / (1.8 * 1000)."""
    return float(np.sum(SOSAT * VOL * np.asarray(kla)) / (AE_SPEC * 1000.0))


def pumping_energy(q_intr, q_r, q_w):
    return float(q_intr * PF_QINTR + q_r * PF_QR + q_w * PF_QW)


def mixing_energy(kla):
    return float(ME_FACTOR * np.sum(VOL * (np.asarray(kla) < 20.0)))


def energy_total(kla, q_intr, q_r, q_w):
    ae = aeration_energy(kla)
    pe = pumping_energy(q_intr, q_r, q_w)
    me = mixing_energy(kla)
    return ae, pe, me, ae + pe + me


def indirect_co2(kwh):
    return kwh * EF_GRID


# ----------------------------------------------------------- effluent quality
def effluent_quantities(eff, asm1par=PAR):
    """Return dict with the BSM1 effluent quality components (g/m3) and flows."""
    nkj = (eff[SNH] + eff[SND] + eff[XND] + asm1par[17] * (eff[XBH] + eff[XBA])
           + asm1par[18] * (eff[XP] + eff[XI]))
    totn = nkj + eff[SNO]
    cod = eff[SS] + eff[SI] + eff[XS] + eff[XI] + eff[XBH] + eff[XBA] + eff[XP]
    bod5 = 0.65 * (eff[SS] + eff[XS] + (1 - asm1par[16]) * (eff[XBH] + eff[XBA]))
    return dict(SNH=eff[SNH], SNO=eff[SNO], NKJ=nkj, TN=totn, COD=cod, BOD5=bod5, TSS=eff[TSS], Q=eff[Q])


def eqi(eff, asm1par=PAR):
    """BSM1 Effluent Quality Index [kg pollution units / d]."""
    q = effluent_quantities(eff, asm1par)
    return (2.0 * q['TSS'] + 1.0 * q['COD'] + 30.0 * q['NKJ'] + 10.0 * q['SNO']
            + 2.0 * q['BOD5']) * q['Q'] / 1000.0
