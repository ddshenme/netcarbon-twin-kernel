# -*- coding: utf-8 -*-
"""Digital-twin layer: steady-state surrogate, time-of-use tariff and dynamic grid factor.

SURROGATE
---------
The fine steady-state map (_sweep_fine.csv) is the twin's predictor: for any admissible
pair of setpoints (dissolved-oxygen setpoint in reactors 3/4/5, nitrate setpoint in
reactor 2) it gives the converged values of
    * effluent NH4-N          [gN/m3]   -> water-quality safety
    * EQI                     [kg/d]    -> water-quality objective
    * electricity             [kWh/d]   -> energy objective
    * direct N2O              [kgCO2e/d]-> carbon objective
The map is built by running the validated plant model to steady state on the same
influent, so the surrogate inherits the model's physics instead of fitting a black box.

Because the map is built at one reference load, the predictors are rescaled by a
first-order load normalisation factor f = (Q*TKN)/(Q_ref*TKN_ref). The scaling is a
documented approximation; the safety margin of the barrier filter covers its error, and
the empirical violation count is reported for every arm.

TARIFF
------
Time-of-use industrial tariff following the structure of 发改价格[2021]1093:
  valley 23:00-08:00 = 0.32 CNY/kWh
  peak   10:00-12:00 and 17:00-21:00 = 1.05 CNY/kWh
  flat   the remaining hours = 0.65 CNY/kWh

DYNAMIC GRID FACTOR
-------------------
The 2023 national average factor is 0.5306 kgCO2/kWh (MEE/NBS announcement 2025 No.47).
Its diurnal shape is modelled from clear-sky solar geometry at 32 N: PV output peaks at
midday, so the grid mix is cleaner then. The shape is normalised to zero daily mean, so
the daily-average factor stays exactly 0.5306 and scenarios remain comparable.
"""
import numpy as np

EF_BASE = 0.5306          # kgCO2/kWh, MEE/NBS 2023 national average
CARBON_PRICE = 0.08       # CNY/kgCO2e  (= 80 CNY/tCO2e, CEA/CCER order of magnitude)
LAT_DEG = 32.0
PV_PENETRATION = 0.15     # midday grid factor is 15 percent below the daily mean
SUN_MEAN = 0.153856       # daily mean of the clear-sky shape at 32 N (winter reference day)


def tou_price(t_days):
    """Time-of-use tariff in CNY/kWh for a time given in days."""
    h = (t_days * 24.0) % 24.0
    if h < 8.0 or h >= 23.0:
        return 0.32
    if (10.0 <= h < 12.0) or (17.0 <= h < 21.0):
        return 1.05
    return 0.65


def pv_shape(t_days):
    """Clear-sky PV generation shape (0 at night) from solar geometry at LAT_DEG."""
    h = (t_days * 24.0) % 24.0
    doy = 1.0 + (t_days % 365.0)
    decl = np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)
    lat = np.radians(LAT_DEG)
    omega = np.radians(15.0 * (h - 12.0))
    sin_alt = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(omega)
    return max(0.0, sin_alt)


_MEAN_CACHE = {}


def _day_mean_shape(t_days):
    """Daily mean of the clear-sky shape for the day containing t_days (cached)."""
    d = int(t_days)
    v = _MEAN_CACHE.get(d)
    if v is None:
        v = float(np.mean([pv_shape(d + i / 96.0) for i in range(96)]))
        if v <= 1e-9:
            v = SUN_MEAN
        _MEAN_CACHE[d] = v
    return v


ORGANIC_N_EFF = 1.9   # effluent organic + particulate N, gN/m3 (from the validated model)


def _cload(x, f_load):
    """First-order load response of an effluent concentration (documented approximation)."""
    return x * (1.0 + 0.8 * (f_load - 1.0)) + 0.8 * (f_load - 1.0)


def grid_factor(t_days):
    """Dynamic grid CO2 factor (kgCO2/kWh); its daily mean equals EF_BASE exactly.

    Midday is cleaner because photovoltaics displace the marginal fossil generator, so
    the factor dips around solar noon and rises at night. Only the *shape* is modelled;
    the level is anchored to the published annual average.
    """
    s = pv_shape(t_days)
    m = _day_mean_shape(t_days)
    return EF_BASE * (1.0 + PV_PENETRATION * (m - s) / m)


# ----------------------------------------------------------------------------- twin
class Twin:
    """Bilinear steady-state surrogate over (DO setpoint, nitrate setpoint)."""

    def __init__(self, csv_path, so_cand, sno_cand):
        import csv as _csv
        rows = list(_csv.DictReader(open(csv_path)))
        so = np.array([float(r['so']) for r in rows])
        sno = np.array([float(r['sno']) for r in rows])
        self.so_grid = np.unique(so)
        self.sno_grid = np.unique(sno)
        ns, nn = self.so_grid.size, self.sno_grid.size
        is_ = np.searchsorted(self.so_grid, so)
        in_ = np.searchsorted(self.sno_grid, sno)
        def grid(key):
            g = np.zeros((ns, nn))
            g[is_, in_] = np.array([float(r[key]) for r in rows])
            return g
        self.KWH = grid('kwh')
        self.EQI = grid('eqi')
        self.DIR = grid('dir')
        self.IND = grid('ind')
        self.SNH = grid('snh')
        self.SNOE = grid('snoe')
        self.TSSE = grid('tss')
        self.EF = grid('ef')
        self.ref_tn = None            # set by the caller (kgN/d at the reference load)
        self.set_candidates(so_cand, sno_cand)

    def set_candidates(self, so_cand, sno_cand):
        self.so_c = np.asarray(so_cand, dtype=float)
        self.sno_c = np.asarray(sno_cand, dtype=float)
        i = np.clip(np.searchsorted(self.so_grid, self.so_c) - 1, 0, self.so_grid.size - 2)
        j = np.clip(np.searchsorted(self.sno_grid, self.sno_c) - 1, 0, self.sno_grid.size - 2)
        self.i = i
        self.j = j
        self.ws = np.clip((self.so_c - self.so_grid[i]) / (self.so_grid[i + 1] - self.so_grid[i]), 0.0, 1.0)
        self.wn = np.clip((self.sno_c - self.sno_grid[j]) / (self.sno_grid[j + 1] - self.sno_grid[j]), 0.0, 1.0)
        self.ncand = self.so_c.size

    def _mix(self, G):
        i, j, ws, wn = self.i, self.j, self.ws, self.wn
        return ((1 - ws) * (1 - wn) * G[i, j] + ws * (1 - wn) * G[i + 1, j]
                + (1 - ws) * wn * G[i, j + 1] + ws * wn * G[i + 1, j + 1])

    def predict(self, f_load):
        """All candidates at once -> dict of arrays (per day at the current load)."""
        kwh = self._mix(self.KWH) * f_load
        eqi = self._mix(self.EQI) * f_load
        dir_ = self._mix(self.DIR) * f_load
        ind = self._mix(self.IND) * f_load
        snh = _cload(self._mix(self.SNH), f_load)
        snoe = _cload(self._mix(self.SNOE), f_load)
        tsse = _cload(self._mix(self.TSSE), f_load)
        return dict(kwh=kwh, eqi=eqi, direct=dir_, indirect=ind, snh=snh, snoe=snoe,
                    tss=tsse, tn=snh + snoe + ORGANIC_N_EFF,
                    so=self.so_c, sno=self.sno_c)

    def one(self, so, sno, f_load):
        """Single-point prediction; convenience wrapper around predict()."""
        self.set_candidates([so], [sno])
        p = self.predict(f_load)
        return {k: (float(v[0]) if hasattr(v, '__len__') else float(v)) for k, v in p.items()}
