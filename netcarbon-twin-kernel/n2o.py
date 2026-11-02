# -*- coding: utf-8 -*-
"""Two-step nitrification (AOB/NOB) + N2O emission module: a metabolic overlay on the validated BSM1 kernel.

WHY AN OVERLAY
--------------
ASM1 lumps all nitrifiers into a single pool (X_BA) and converts NH4+ straight to NO3-,
which makes nitrite (NO2-) invisible. Nitrite is the single most important precursor of
N2O in full-scale plants. Rewriting the (already cross-validated) ASM1 kernel would
invalidate that validation, so this module instead *reads* the ASM1 state trajectory,
resolves the nitrogen flux through the two-step pathway, and carries NO2- and N2O as
extra state variables. It does not feed back into the plant mass balance.

Nitrogen is conserved by construction: the ammonia-oxidation flux J_AOB is taken to be
exactly ASM1's autotrophic nitrogen consumption, i.e. J_AOB = (1/Y_A + I_XB) * p3, so
the total nitrification flux is unchanged. The overlay only decides how much of the
intermediate NO2- is left unconsumed and how much of it leaves as N2O.

MECHANISMS (standard, cited in the report)
------------------------------------------
1. Differential oxygen affinity: AOB have a lower O2 half-saturation than NOB
   (K_OA=0.40 vs K_OA_NOB=2.00 mg/L). Below DO ~ 1.5 mg/L the second step becomes
   O2-limited and NO2- accumulates.                     [Blackburne et al. 2008]
2. Free-ammonia (FA) inhibition of NOB, stronger than for AOB.
   [Anthonisen et al. 1976: 0.1-1.0 mg NH3-N/L inhibits NOB]. pH is assumed constant
   at 7.0 because BSM1 carries no pH state -- documented modelling limitation.
3. N2O from three pathways [Pocquet et al. 2016 two-pathway AOB model + heterotrophic]:
     r_NN  AOB denitrification    - AOB active, low O2, driven by NO2-
     r_ND  hydroxylamine pathway  - AOB active and O2 present
     r_HD  heterotrophic incomplete denitrification - anoxic, driven by NO3-
4. N2O is stripped by aeration (kLa_N2O = 0.9 * kLa_O2) and leaves dissolved in the
   effluent. The emission factor is reported against the influent total-N load, as in
   Massara et al. (SWWS) -- including both stripped and dissolved N2O.

STATE LAYOUT (NEXT = 255 = 225 ASM1 + 30 overlay)
  [225:230] S_NO2  reactors 1..5      (gN/m3)
  [230:235] S_N2O  reactors 1..5      (gN/m3)
  [235:245] S_NO2  settler layers 1..10
  [245:255] S_N2O  settler layers 1..10
"""
import numpy as np
from numba import njit
from bsm1_fast import (SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK,
                       TSS, Q, TEMP, MU_A, K_NH, K_OA, Y_A, I_XB,
                       plant_deriv_nb, settler_output_nb, VOL1, VOL2, VOL3,
                       NOFLAYERS, FEEDLAYER, AREA, HEIGHT, I_XP)

NEXT = 255
OFF_NO2_R = 225
OFF_N2O_R = 230
OFF_NO2_S = 235
OFF_N2O_S = 245

# ---- two-step nitrification (calibrated; see report section on DO response) ----
MU_NOB = 2.00          # effective lumped NOB capacity (NOB biomass is not separate in ASM1)
K_NO2_NOB = 1.00       # gN/m3, NO2- half-saturation for NOB
K_OA_NOB = 2.00        # gO2/m3, NOB O2 half-saturation (vs K_OA=0.40 for AOB)
K_I_FA_NOB = 0.20      # gN/m3, free-ammonia inhibition constant for NOB
PH_ASSUMED = 7.0

# ---- N2O production pathways (fractions of the AOB ammonia-oxidation flux) -----
K_NN = 0.100           # max fraction of J_AOB emitted as N2O via AOB denitrification
K_O_NN = 0.40          # gO2/m3, O2 half-saturation for AOB denitrification
K_I_NN = 0.30          # gO2/m3, O2 inhibition constant (Ni et al. 2013: 0.1-0.5)
K_NO2_NN = 0.50        # gN/m3, NO2- half-saturation for AOB denitrification
K_ND = 0.010           # max fraction of J_AOB emitted as N2O via the NH2OH pathway
K_O_ND = 0.50          # gO2/m3
K_NH_ND = 1.00         # gN/m3
K_HD = 0.0005          # heterotrophic incomplete denitrification, 1/d
K_I_O2_HD = 0.20       # gO2/m3
K_NO_HD = 0.50         # gN/m3

# ---- N2O physical transfer -----------------------------------------------------
KL_N2O_RATIO = 0.90    # kLa(N2O) / kLa(O2), from the diffusivity ratio
C_SAT_N2O = 0.0003     # gN/m3

# ---- global warming potentials (IPCC AR6) --------------------------------------
GWP_N2O = 273.0
GWP_CH4 = 27.0

# ---- CH4 diagnostic (see report: BSM1 water line has no strictly anaerobic zone) -
K_CH4 = 0.010          # 1/d, first-order on readily biodegradable COD
K_I_O2_CH4 = 0.10      # gO2/m3
K_I_NO3_CH4 = 0.20     # gN/m3, nitrate inhibition of methanogenesis

VOLS = (VOL1, VOL2, VOL3, VOL3, VOL3)
AOB_STOICH = 1.0 / Y_A + I_XB     # 4.25: N oxidised per unit of autotrophic growth


@njit(cache=True, fastmath=True)
def _fa_fraction(t_c):
    """Fraction of total ammonia present as free NH3 at pH 7.0 (pKa from temperature)."""
    pka = 0.09018 + 2729.92 / (273.15 + t_c)
    return 1.0 / (1.0 + 10.0 ** (pka - PH_ASSUMED))


@njit(cache=True, fastmath=True)
def _rates(so, snh, sno, no2, xba, xbh, t_c):
    """Return (J_AOB, r_nob, r_nn, r_nd, r_hd). All are gN/m3/d nitrogen fluxes."""
    p3 = MU_A * (snh / (K_NH + snh)) * (so / (K_OA + so)) * xba
    j_aob = AOB_STOICH * p3
    fa = snh * _fa_fraction(t_c)
    inh_fa = K_I_FA_NOB / (K_I_FA_NOB + fa)
    r_nob = (AOB_STOICH * MU_NOB * (no2 / (K_NO2_NOB + no2))
             * (so / (K_OA_NOB + so)) * inh_fa * xba)
    r_nn = (K_NN * (so / (K_O_NN + so)) * (K_I_NN / (K_I_NN + so))
            * (no2 / (K_NO2_NN + no2)) * j_aob)
    r_nd = K_ND * (so / (K_O_ND + so)) * (snh / (K_NH_ND + snh)) * j_aob
    r_hd = K_HD * (K_I_O2_HD / (K_I_O2_HD + so)) * (sno / (K_NO_HD + sno)) * xbh
    return j_aob, r_nob, r_nn, r_nd, r_hd


@njit(cache=True, fastmath=True)
def settler_soluble_deriv(c, c_in, q_feed, q_r, q_w):
    """Transport of a non-settling soluble solute through the 10-layer Takacs settler."""
    n = NOFLAYERS
    h = HEIGHT / n
    v_in = q_feed / AREA
    q_u = q_r + q_w
    v_up = (q_feed - q_u) / AREA
    v_dn = q_u / AREA
    dc = np.zeros(n)
    for i in range(FEEDLAYER - 1):
        dc[i] = (-v_up * c[i] + v_up * c[i + 1]) / h
    f = FEEDLAYER - 1
    dc[f] = (v_in * c_in - v_up * c[f] - v_dn * c[f]) / h
    for i in range(FEEDLAYER, n):
        dc[i] = (v_dn * c[i - 1] - v_dn * c[i]) / h
    return dc


@njit(cache=True, fastmath=True)
def plant_deriv_ext(y, t, inf, u):
    dy = np.zeros(NEXT)
    base = plant_deriv_nb(y[:225], t, inf, u)
    for i in range(225):
        dy[i] = base[i]

    kla = u[0:5]
    q_intr, q_r, q_w = u[5], u[6], u[7]
    q_in = inf[Q]
    yr = y[:105].reshape(5, 21)
    ys = y[105:225]

    y_feed = yr[4].copy()
    y_feed[Q] = q_in + q_r
    ret, eff = settler_output_nb(ys, y_feed, q_r, q_w)

    q1 = q_in + q_r + q_intr
    no2_in = (y[OFF_NO2_S + NOFLAYERS - 1] * q_r + y[OFF_NO2_R + 4] * q_intr) / q1
    n2o_in = (y[OFF_N2O_S + NOFLAYERS - 1] * q_r + y[OFF_N2O_R + 4] * q_intr) / q1

    for i in range(5):
        so, snh, sno = yr[i][SO], yr[i][SNH], yr[i][SNO]
        no2 = y[OFF_NO2_R + i]
        n2o = y[OFF_N2O_R + i]
        vol = VOLS[i]
        j_aob, r_nob, r_nn, r_nd, r_hd = _rates(so, snh, sno, no2, yr[i][XBA], yr[i][XBH], yr[i][TEMP])
        dy[OFF_NO2_R + i] = (j_aob - r_nob) + (q1 * (no2_in - no2)) / vol
        dy[OFF_N2O_R + i] = (r_nn + r_nd + r_hd + KL_N2O_RATIO * kla[i] * (C_SAT_N2O - n2o)
                             + (q1 * (n2o_in - n2o)) / vol)
        no2_in = no2
        n2o_in = n2o

    d_no2 = settler_soluble_deriv(y[OFF_NO2_S:OFF_NO2_S + NOFLAYERS],
                                  y[OFF_NO2_R + 4], q_in + q_r, q_r, q_w)
    d_n2o = settler_soluble_deriv(y[OFF_N2O_S:OFF_N2O_S + NOFLAYERS],
                                  y[OFF_N2O_R + 4], q_in + q_r, q_r, q_w)
    for i in range(NOFLAYERS):
        dy[OFF_NO2_S + i] = d_no2[i]
        dy[OFF_N2O_S + i] = d_n2o[i]
    return dy


@njit(cache=True, fastmath=True)
def rk4_step_ext(y, dt, inf, u, nsub):
    h = dt / nsub
    for _ in range(nsub):
        k1 = plant_deriv_ext(y, 0.0, inf, u)
        k2 = plant_deriv_ext(y + 0.5 * h * k1, 0.0, inf, u)
        k3 = plant_deriv_ext(y + 0.5 * h * k2, 0.0, inf, u)
        k4 = plant_deriv_ext(y + h * k3, 0.0, inf, u)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for j in range(NEXT):
            if y[j] < 0.0:
                y[j] = 0.0
    return y


@njit(cache=True, fastmath=True)
def n2o_fluxes(y, inf, u):
    """Instantaneous N2O nitrogen fluxes (gN/d):
    (stripped, effluent, wasted, produced, from_nn, from_nd, from_hd)."""
    kla = u[0:5]
    q_in, q_w = inf[Q], u[7]
    yr = y[:105].reshape(5, 21)
    stripped = 0.0
    prod = 0.0
    f_nn = 0.0
    f_nd = 0.0
    f_hd = 0.0
    for i in range(5):
        _a, r_nob, r_nn, r_nd, r_hd = _rates(yr[i][SO], yr[i][SNH], yr[i][SNO],
                                             y[OFF_NO2_R + i], yr[i][XBA], yr[i][XBH], yr[i][TEMP])
        prod += (r_nn + r_nd + r_hd) * VOLS[i]
        f_nn += r_nn * VOLS[i]
        f_nd += r_nd * VOLS[i]
        f_hd += r_hd * VOLS[i]
        s = KL_N2O_RATIO * kla[i] * (y[OFF_N2O_R + i] - C_SAT_N2O)
        if s > 0.0:
            stripped += s * VOLS[i]
    eff_n2o = y[OFF_N2O_S] * (q_in - q_w)
    was_n2o = y[OFF_N2O_S + NOFLAYERS - 1] * q_w
    return stripped, eff_n2o, was_n2o, prod, f_nn, f_nd, f_hd


@njit(cache=True, fastmath=True)
def nitrite_profile(y):
    """S_NO2 in the 5 reactors and in the 10 settler layers (for reporting)."""
    out = np.zeros(15)
    for i in range(5):
        out[i] = y[OFF_NO2_R + i]
    for i in range(NOFLAYERS):
        out[5 + i] = y[OFF_NO2_S + i]
    return out


@njit(cache=True, fastmath=True)
def ch4_flux(y, inf, u):
    """Diagnostic CH4 flux (gN-equivalent/d) from the water line.

    BSM1 has no strictly anaerobic zone and its anoxic tanks carry nitrate, so both O2
    and NO3- inhibit methanogenesis. The sludge line (thickener + digester) is outside
    the BSM1 boundary and is therefore not modelled; this is stated in the report.
    """
    yr = y[:105].reshape(5, 21)
    q_in = inf[Q]
    tot = 0.0
    for i in range(5):
        so = yr[i][SO]
        sno = yr[i][SNO]
        ss = yr[i][SS]
        r = K_CH4 * (K_I_O2_CH4 / (K_I_O2_CH4 + so)) * (K_I_NO3_CH4 / (K_I_NO3_CH4 + sno)) * ss
        tot += r * VOLS[i]
    return tot


@njit(cache=True, fastmath=True)
def init_n2o(y):
    """Append the 30 overlay states to a converged 225-state ASM1 plant state."""
    y2 = np.zeros(NEXT)
    for i in range(225):
        y2[i] = y[i]
    for i in range(5):
        y2[OFF_NO2_R + i] = 0.20
        y2[OFF_N2O_R + i] = 0.001
    for i in range(NOFLAYERS):
        y2[OFF_NO2_S + i] = 0.20
        y2[OFF_N2O_S + i] = 0.001
    return y2


@njit(cache=True, fastmath=True)
def influent_tn(inf):
    nkj = (inf[SNH] + inf[SND] + inf[XND] + I_XB * (inf[XBH] + inf[XBA])
           + I_XP * (inf[XP] + inf[XI]))
    return nkj + inf[SNO]


@njit(cache=True, fastmath=True)
def n2o_ef(y, inf, u):
    """Emission factor: N2O-N released (stripped + effluent + wasted) / influent TN, as a fraction."""
    stripped, eff, was, prod, a, b, c = n2o_fluxes(y, inf, u)
    return (stripped + eff + was) / (influent_tn(inf) * inf[Q])
