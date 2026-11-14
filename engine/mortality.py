"""Mortality basis: Gompertz-Makeham, select factors, smoker and HNW adjustments."""
from __future__ import annotations

import numpy as np


def makeham_q(ages: np.ndarray, A: float, B: float, c: float) -> np.ndarray:
    """One year death probability from the Makeham force mu(x) = A + B c^x.

    q_x = 1 - exp(-(A + B c^x (c - 1) / ln c))
    """
    ages = np.asarray(ages, dtype=float)
    integral = A + B * c ** ages * (c - 1.0) / np.log(c)
    return 1.0 - np.exp(-integral)


class MortalityBasis:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def base_q(self, ages, sex: str, unisex: bool) -> np.ndarray:
        m = self.cfg["male"]
        f = self.cfg["female"]
        qm = makeham_q(ages, m["A"], m["B"], m["c"])
        qf = makeham_q(ages, f["A"], f["B"], f["c"])
        if unisex:
            w = self.cfg["unisex_male_weight"]
            return w * qm + (1.0 - w) * qf
        return qm if sex == "M" else qf

    def select_vector(self, term: int) -> np.ndarray:
        sel = list(self.cfg["select_factors"])
        out = np.ones(term)
        out[: min(term, len(sel))] = sel[:term]
        return out

    def standard_q(self, entry_age: int, term: int, sex: str, smoker: bool, unisex: bool) -> np.ndarray:
        """Standard insured HNW basis by policy year, before individual ratings."""
        ages = entry_age + np.arange(term)
        q = self.base_q(ages, sex, unisex)
        q = q * self.select_vector(term) * self.cfg["hnw_factor"]
        if smoker:
            q = q * self.cfg["smoker_factor"]
        return q


def rated_q(standard_q: np.ndarray, debit_pct: float, flat_extra: np.ndarray, multiplier: float = 1.0,
            cap: float = 1.0) -> np.ndarray:
    """Apply a percentage debit and a flat extra (per unit, not per mille)."""
    q = standard_q * multiplier * (1.0 + debit_pct / 100.0) + flat_extra
    return np.clip(q, 0.0, cap)
