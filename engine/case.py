"""Pricing, reinsurance, risk and capital for one case.

Every result is built on one exact outcome model of the single policy. The policy ends in
exactly one of 2n + 1 ways: death in year t, lapse at the end of year t, or survival to
expiry. Each outcome has a probability and a present value of cash flows, gross and net of
reinsurance. Expected values, the premium, the exact profit distribution, the Monte Carlo
simulation and the capital shocks are all computed from that single model, so they are
consistent with each other by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import Applicant, Policy, ReinsuranceProgram
from .mortality import MortalityBasis, rated_q
from .underwriting import UWResult


# ----------------------------------------------------------------------------- helpers

def outcome_probs(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Probabilities of [death y0..y(n-1), lapse y0..y(n-1), survival]. Works on 1D or 2D (sims x years)."""
    q = np.atleast_2d(q)
    w = np.broadcast_to(w, q.shape)
    stay = (1.0 - q) * (1.0 - w)
    l = np.concatenate([np.ones((q.shape[0], 1)), np.cumprod(stay, axis=1)], axis=1)  # in force at start of year
    death = l[:, :-1] * q
    lapse = l[:, :-1] * (1.0 - q) * w
    surv = l[:, -1:]
    out = np.concatenate([death, lapse, surv], axis=1)
    return out[0] if out.shape[0] == 1 else out


def in_force(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    stay = (1.0 - q) * (1.0 - w)
    return np.concatenate([[1.0], np.cumprod(stay)])


# ----------------------------------------------------------------------------- results

@dataclass
class Layer:
    name: str
    kind: str              # "Retention", "Treaty", "Facultative"
    rating: str
    sum_assured: float
    rate_pct: float


@dataclass
class CaseResult:
    premium: float = 0.0
    premium_mode: str = "annual"
    rate_permille: float = 0.0
    decomposition: dict = field(default_factory=dict)
    cashflows: pd.DataFrame | None = None
    layers: list[Layer] = field(default_factory=list)
    reinsurance_summary: dict = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    outcomes: pd.DataFrame | None = None
    exact_stats: dict = field(default_factory=dict)
    capital: dict = field(default_factory=dict)


class CaseModel:
    def __init__(self, app: Applicant, pol: Policy, uw: UWResult, ri: ReinsuranceProgram, cfg: dict):
        self.app, self.pol, self.uw, self.ri, self.cfg = app, pol, uw, ri, cfg
        n = pol.term
        self.n = n
        self.i = cfg["pricing"]["interest_rate"]
        self.v = 1.0 / (1.0 + self.i)
        self.t = np.arange(n)

        basis = MortalityBasis(cfg["mortality"])
        self.q_std = basis.standard_q(app.age, n, app.sex, app.smoker, pol.unisex)
        self.flat = uw.flat_extra_vector(n)
        self.q = rated_q(self.q_std, uw.total_debit, self.flat)

        if pol.premium_mode == "single":
            self.w = np.zeros(n)
        else:
            lr = cfg["pricing"]["lapse_rates"]
            self.w = np.array([lr[min(k, len(lr) - 1)] for k in range(n)], dtype=float)
            self.w[-1] = 0.0  # lapse in the final year is indistinguishable from expiry

        self._build_layers()

    # ------------------------------------------------------------------ reinsurance layers
    def _build_layers(self) -> None:
        sa, ri = self.pol.sum_assured, self.ri
        retained = min(sa, ri.retention)
        treaty = min(max(sa - ri.retention, 0.0), ri.retention * ri.surplus_lines)
        fac = max(sa - retained - treaty, 0.0)
        self.layers = [Layer("Company retention", "Retention", "", retained, 0.0)]
        if treaty > 0:
            self.layers.append(Layer("Surplus treaty", "Treaty", "", treaty, ri.treaty_rate_pct))
        self.alerts: list[str] = []
        if fac > 0:
            total_share = sum(r.share for r in ri.facultative_pool)
            for r in ri.facultative_pool:
                self.layers.append(Layer(r.name, "Facultative", r.rating, fac * r.share, r.rate_pct))
            if abs(total_share - 1.0) > 1e-9:
                gap = fac * (1.0 - total_share)
                self.layers[0].sum_assured += gap
                if gap > 0:
                    self.alerts.append(f"Facultative pool placed at {total_share:.0%}. "
                                       f"Unplaced EUR {gap:,.0f} falls back to the company's retention.")
                else:
                    self.alerts.append(f"Facultative pool is over-placed at {total_share:.0%}. "
                                       "Shares should sum to 100%.")
        fac_total = sum(l.sum_assured for l in self.layers if l.kind == "Facultative")
        for l in self.layers:
            if l.kind == "Facultative" and fac_total > 0 and l.sum_assured / fac_total > ri.concentration_alert:
                self.alerts.append(f"Concentration: {l.name} carries {l.sum_assured / fac_total:.0%} of the "
                                   "facultative layer.")
        if self.layers[0].sum_assured > ri.retention + 1e-6:
            self.alerts.append("Retained sum assured exceeds the company's retention limit.")

    @property
    def ceded_sa(self) -> float:
        return sum(l.sum_assured for l in self.layers if l.kind != "Retention")

    # ------------------------------------------------------------------ cash flow components
    def _components(self, premium: float):
        """Return per-year PV components (as of time 0) used by the outcome model."""
        c = self.cfg["pricing"]
        n, t, v = self.n, self.t, self.v
        disc_start = v ** t
        disc_end = v ** (t + 1)

        if self.pol.premium_mode == "single":
            prem = np.zeros(n); prem[0] = premium
            comm = prem * c["single_premium_commission"]
        else:
            prem = np.full(n, premium)
            comm = prem * c["renewal_commission"]
            comm[0] = premium * c["initial_commission"]
        exp = c["renewal_expense"] * (1 + c["expense_inflation"]) ** t
        exp = exp.astype(float)
        exp[0] = c["initial_expense"] + self.uw.underwriting_cost

        gross_start = (prem - comm - exp) * disc_start

        # Reinsurance premiums (YRT on ceded sum assured, paid at the start of each year)
        ri_prem = np.zeros(n)
        allowance = np.zeros(n)
        brokerage = np.zeros(n)
        per_layer = {}
        for l in self.layers:
            if l.kind == "Retention" or l.sum_assured <= 0:
                continue
            q_re = rated_q(self.q_std * l.rate_pct, self.uw.total_debit, self.flat)
            p = l.sum_assured * q_re
            per_layer[l.name] = p
            ri_prem += p
            if l.kind == "Treaty":
                allowance[0] += p[0] * self.ri.treaty_first_year_allowance
                allowance[1:] += p[1:] * self.ri.treaty_renewal_allowance
            else:
                brokerage += p * self.ri.brokerage
        ri_start = (-ri_prem + allowance - brokerage) * disc_start

        claim_end = -self.pol.sum_assured * disc_end
        recovery_end = self.ceded_sa * disc_end
        return dict(prem=prem, comm=comm, exp=exp, ri_prem=ri_prem, allowance=allowance, brokerage=brokerage,
                    per_layer=per_layer, gross_start=gross_start, ri_start=ri_start,
                    claim_end=claim_end, recovery_end=recovery_end, disc_start=disc_start, disc_end=disc_end)

    def outcome_values(self, premium: float) -> tuple[np.ndarray, np.ndarray]:
        """PV of profit for each of the 2n+1 outcomes, gross and net of reinsurance."""
        k = self._components(premium)
        cg = np.cumsum(k["gross_start"])
        cr = np.cumsum(k["ri_start"])
        death_g = cg + k["claim_end"]
        death_n = death_g + cr + k["recovery_end"]
        lapse_g, lapse_n = cg, cg + cr
        gross = np.concatenate([death_g, lapse_g, cg[-1:]])
        net = np.concatenate([death_n, lapse_n, (cg + cr)[-1:]])
        return gross, net

    # ------------------------------------------------------------------ pricing
    def price(self, net: bool = False) -> float:
        """Solve E[PV profit] = margin * PV(premiums), gross or net of reinsurance.
        PV profit is linear in the premium, so two evaluations give the exact solution."""
        m = self.cfg["pricing"]["profit_margin"]
        p = outcome_probs(self.q, self.w)
        g0 = self.outcome_values(0.0)[1 if net else 0]
        g1 = self.outcome_values(1.0)[1 if net else 0]
        a = p @ g0
        b = p @ g1 - a
        l = in_force(self.q, self.w)[:-1]
        annuity = 1.0 if self.pol.premium_mode == "single" else float(l @ (self.v ** self.t))
        return -a / (b - m * annuity)

    # ------------------------------------------------------------------ full run
    def run(self) -> CaseResult:
        res = CaseResult(premium_mode=self.pol.premium_mode)
        P = self.price()
        res.premium = P
        res.rate_permille = 1000 * P / self.pol.sum_assured
        res.layers = self.layers
        res.alerts = list(self.alerts)

        k = self._components(P)
        l = in_force(self.q, self.w)
        lt = l[:-1]
        probs = outcome_probs(self.q, self.w)
        gross, net = self.outcome_values(P)

        pv_prem = float(lt @ (k["prem"] * k["disc_start"]))
        pv_comm = float(lt @ (k["comm"] * k["disc_start"]))
        pv_exp = float(lt @ (k["exp"] * k["disc_start"]))
        pv_claims = float(lt @ (self.q * self.pol.sum_assured * k["disc_end"]))
        pv_profit_g = float(probs @ gross)
        res.decomposition = {
            "Expected claims": pv_claims, "Expenses and underwriting": pv_exp,
            "Distribution commission": pv_comm, "Profit margin": pv_profit_g,
            "PV premiums": pv_prem,
        }

        pv_ri_prem = float(lt @ (k["ri_prem"] * k["disc_start"]))
        pv_allow = float(lt @ (k["allowance"] * k["disc_start"]))
        pv_brok = float(lt @ (k["brokerage"] * k["disc_start"]))
        pv_recov = float(lt @ (self.q * self.ceded_sa * k["disc_end"]))
        pv_profit_n = float(probs @ net)
        res.reinsurance_summary = {
            "PV reinsurance premiums": pv_ri_prem, "PV treaty allowances": pv_allow,
            "PV brokerage": pv_brok, "PV expected recoveries": pv_recov,
            "Net cost of reinsurance": pv_ri_prem - pv_allow + pv_brok - pv_recov,
            "PV profit gross": pv_profit_g, "PV profit net": pv_profit_n,
            "Ceded sum assured": self.ceded_sa,
            "Retained sum assured": self.layers[0].sum_assured,
            "Premium for target margin net of reinsurance": self.price(net=True),
        }
        res.reinsurance_summary["Per layer PV premium"] = {
            name: float(lt @ (p * k["disc_start"])) for name, p in k["per_layer"].items()}

        years = self.t + 1
        res.cashflows = pd.DataFrame({
            "Policy year": years, "Attained age": self.app.age + self.t,
            "q standard (per mille)": 1000 * self.q_std, "q rated (per mille)": 1000 * self.q,
            "In force": lt, "Premium": k["prem"] * lt, "Commission": k["comm"] * lt,
            "Expenses": k["exp"] * lt, "Expected claims": self.q * self.pol.sum_assured * lt,
            "Reinsurance premium": k["ri_prem"] * lt, "Allowance": k["allowance"] * lt,
            "Brokerage": k["brokerage"] * lt, "Expected recoveries": self.q * self.ceded_sa * lt,
        })

        labels = ([f"Death in year {y}" for y in years] + [f"Lapse after year {y}" for y in years]
                  + ["Survival to expiry"])
        res.outcomes = pd.DataFrame({"Outcome": labels, "Probability": probs, "PV profit gross": gross,
                                     "PV profit net": net})
        res.outcomes = res.outcomes[res.outcomes["Probability"] > 0].reset_index(drop=True)
        res.exact_stats = {"gross": distribution_stats(gross, probs), "net": distribution_stats(net, probs)}

        res.capital = self.capital(P)
        return res

    # ------------------------------------------------------------------ capital
    def capital(self, premium: float) -> dict:
        cc = self.cfg["capital"]
        gross, net = self.outcome_values(premium)
        p0 = outcome_probs(self.q, self.w)
        q_shock = np.clip(self.q * (1 + cc["mortality_shock"]), 0, 1)
        p1 = outcome_probs(q_shock, self.w)
        # Loss of value under the shock = increase in best estimate liability
        mort_g = max(float(p0 @ gross - p1 @ gross), 0.0)
        mort_n = max(float(p0 @ net - p1 @ net), 0.0)
        cat_g = cc["cat_shock"] * self.pol.sum_assured * self.v
        cat_n = cc["cat_shock"] * self.layers[0].sum_assured * self.v
        rho = cc["corr_mort_cat"]
        life_g = float(np.sqrt(mort_g**2 + cat_g**2 + 2 * rho * mort_g * cat_g))
        life_n = float(np.sqrt(mort_n**2 + cat_n**2 + 2 * rho * mort_n * cat_n))

        # Approximate capital run-off: CAT follows the in-force, mortality also the remaining term
        l = in_force(self.q, self.w)[:-1]
        remaining = (self.n - self.t) / self.n
        def pv_coc(m0, c0):
            m_t, c_t = m0 * l * remaining, c0 * l
            scr_t = np.sqrt(m_t**2 + c_t**2 + 2 * rho * m_t * c_t)
            return float(cc["cost_of_capital"] * scr_t @ (self.v ** (self.t + 1)))
        coc_g, coc_n = pv_coc(mort_g, cat_g), pv_coc(mort_n, cat_n)
        return {
            "SCR mortality gross": mort_g, "SCR mortality net": mort_n,
            "SCR catastrophe gross": cat_g, "SCR catastrophe net": cat_n,
            "SCR life gross": life_g, "SCR life net": life_n,
            "Capital relief": life_g - life_n,
            "PV cost of capital gross": coc_g, "PV cost of capital net": coc_n,
            "PV cost of capital saved": coc_g - coc_n,
        }

    # ------------------------------------------------------------------ Monte Carlo
    def simulate(self, premium: float, n_sims: int, portfolio: int, sigma: float, seed: int) -> dict:
        """Portfolio of `portfolio` independent lives with this profile, with a common lognormal
        uncertainty on the mortality level (systematic risk that does not diversify)."""
        rng = np.random.default_rng(seed)
        theta = np.exp(sigma * rng.standard_normal(n_sims) - 0.5 * sigma**2)
        q = np.clip(self.q[None, :] * theta[:, None], 0, 1)
        probs = outcome_probs(q, self.w)
        if probs.ndim == 1:
            probs = probs[None, :]
        cdf = np.cumsum(probs, axis=1)
        gross, net = self.outcome_values(premium)
        K = gross.size
        total_g = np.zeros(n_sims)
        total_n = np.zeros(n_sims)
        claims = np.zeros(n_sims, dtype=int)
        chunk = max(1, 2_000_000 // max(1, portfolio * K))
        for s in range(0, n_sims, chunk):
            e = min(n_sims, s + chunk)
            u = rng.random((e - s, portfolio))
            idx = (u[:, :, None] > cdf[s:e, None, :]).sum(axis=2)
            idx = np.minimum(idx, K - 1)
            total_g[s:e] = gross[idx].sum(axis=1)
            total_n[s:e] = net[idx].sum(axis=1)
            claims[s:e] = (idx < self.n).sum(axis=1)
        return {"theta": theta, "gross": total_g, "net": total_n, "claims": claims,
                "stats_gross": sample_stats(total_g), "stats_net": sample_stats(total_n)}


# ----------------------------------------------------------------------------- statistics

def distribution_stats(values: np.ndarray, probs: np.ndarray, level: float = 0.995) -> dict:
    """Exact statistics of a discrete distribution of PV profit. VaR and TVaR are on losses."""
    loss = -values
    order = np.argsort(loss)
    lv, pv = loss[order], probs[order]
    cdf = np.cumsum(pv)
    var = float(lv[np.searchsorted(cdf, level - 1e-12)])
    tail_p = 1 - level
    # TVaR: mean of the worst (1 - level) probability mass
    rev_l, rev_p = lv[::-1], pv[::-1]
    take = np.minimum(rev_p, np.maximum(tail_p - np.concatenate([[0], np.cumsum(rev_p)[:-1]]), 0))
    tvar = float(rev_l @ take / tail_p)
    mean = float(probs @ values)
    return {"mean": mean, "std": float(np.sqrt(probs @ (values - mean) ** 2)),
            "prob_loss": float(probs[values < 0].sum()), "VaR 99.5% loss": var, "TVaR 99.5% loss": tvar,
            "worst": float(values.min())}


def sample_stats(x: np.ndarray, level: float = 0.995) -> dict:
    loss = -x
    var = float(np.quantile(loss, level))
    tail = loss[loss >= var]
    return {"mean": float(x.mean()), "std": float(x.std()), "prob_loss": float((x < 0).mean()),
            "VaR 99.5% loss": var, "TVaR 99.5% loss": float(tail.mean()), "worst": float(x.min())}
