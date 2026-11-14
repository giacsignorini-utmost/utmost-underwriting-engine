"""Numerical rating underwriting engine.

All ratings are ILLUSTRATIVE. They follow the structure used in life underwriting manuals
(percentage debits and credits on mortality, flat extras in per mille of sum assured,
postpone and decline outcomes) but the values are the author's own assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import Applicant, Policy, UWLine

# Catalogue used by the UI: code -> (label, parameter spec)
CONDITIONS = {
    "diabetes_t2": ("Type 2 diabetes", {"hba1c": ("HbA1c (%)", 5.0, 12.0, 6.8, 0.1)}),
    "hypertension": ("Treated hypertension", {}),
    "coronary": ("Coronary artery disease", {"years_since_event": ("Years since last event", 0, 30, 4, 1)}),
    "cancer": ("Cancer history", {
        "type": ("Type", ["breast", "prostate", "colorectal", "melanoma", "other"]),
        "years_since_remission": ("Years since remission", 0, 30, 6, 1),
    }),
    "sleep_apnea": ("Sleep apnoea", {"treated": ("Treated with CPAP", [True, False])}),
    "mental_health": ("Anxiety or depression", {"severity": ("Severity", ["mild", "moderate", "severe"])}),
    "asthma": ("Asthma", {"severity": ("Severity", ["mild", "moderate", "severe"])}),
    "hepatitis": ("Hepatitis B or C", {"status": ("Status", ["cured", "active"])}),
    "alcohol": ("Alcohol consumption", {"units_per_week": ("Units per week", 0, 100, 25, 1)}),
}

TABLE_LETTERS = "ABCDEFGHIJKLMNOP"  # Table A = +25%, B = +50%, ...
MAX_DEBIT = 300.0                   # above this the case is declined
MAX_FLAT_EXTRA = 10.0               # per mille, above this the case is declined


@dataclass
class UWResult:
    lines: list[UWLine] = field(default_factory=list)
    total_debit: float = 0.0
    risk_class: str = "Standard"
    decision: str = "Accept"          # Accept, Accept rated, Postpone, Decline
    flat_extras: list[tuple[float, int | None]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    financial_flags: list[str] = field(default_factory=list)
    underwriting_cost: float = 0.0

    @property
    def insurable(self) -> bool:
        return self.decision in ("Accept", "Accept rated")

    def flat_extra_vector(self, term: int) -> np.ndarray:
        """Flat extras by policy year, as probabilities (per mille / 1000)."""
        v = np.zeros(term)
        for permille, years in self.flat_extras:
            n = term if years is None else min(term, years)
            v[:n] += permille / 1000.0
        return v

    @property
    def max_flat_extra_permille(self) -> float:
        return sum(p for p, _ in self.flat_extras)


def _band(value, bands):
    """bands: list of (upper_bound_exclusive, result). Last result applies above."""
    for upper, res in bands:
        if value < upper:
            return res
    return bands[-1][1]


def assess(app: Applicant, pol: Policy) -> UWResult:
    res = UWResult()
    L = res.lines

    # Build
    bmi = app.bmi
    if bmi >= 45:
        L.append(UWLine("Build", f"BMI {bmi:.1f}", outcome="decline"))
    else:
        d = _band(bmi, [(17, 75), (18.5, 25), (30, 0), (35, 50), (40, 100), (45, 200)])
        L.append(UWLine("Build", f"BMI {bmi:.1f}", debit_pct=d))

    # Blood pressure
    if app.systolic_bp >= 180:
        L.append(UWLine("Blood pressure", f"Systolic {app.systolic_bp} mmHg", outcome="postpone"))
    else:
        d = _band(app.systolic_bp, [(140, 0), (160, 25), (180, 75)])
        L.append(UWLine("Blood pressure", f"Systolic {app.systolic_bp} mmHg", debit_pct=d))

    # Lipids
    d = _band(app.chol_hdl_ratio, [(6, 0), (8, 25), (99, 50)])
    L.append(UWLine("Lipids", f"Total cholesterol / HDL {app.chol_hdl_ratio:.1f}", debit_pct=d))

    # Smoking: separate basis, shown for transparency
    L.append(UWLine("Tobacco", "Smoker basis" if app.smoker else "Non-smoker basis"))

    # Family history
    if app.family_history:
        L.append(UWLine("Family history", "Parent or sibling CV or cancer death before 60", debit_pct=25))

    # Medical conditions
    for c in app.conditions:
        L.extend(_condition_lines(c.code, c.detail, app.age))

    # Occupation
    occ = {1: (0, "Office or professional"), 2: (0, "Light manual"),
           3: (1.0, "Heavy manual or frequent field travel"), 4: (3.0, "Hazardous (offshore, security)")}
    fe, label = occ[app.occupation_class]
    L.append(UWLine("Occupation", f"Class {app.occupation_class}: {label}", flat_extra_permille=fe))

    # Avocations
    av = app.avocations or {}
    if av.get("pilot_hours", 0) > 0:
        h = av["pilot_hours"]
        L.append(UWLine("Aviation", f"Private pilot, {h} h per year", flat_extra_permille=2.5 if h < 50 else 2.0))
    depth = av.get("dive_depth_m", 0)
    if depth > 0:
        if depth > 60:
            L.append(UWLine("Diving", f"Max depth {depth} m", flat_extra_permille=4.0))
        elif depth > 40:
            L.append(UWLine("Diving", f"Max depth {depth} m", flat_extra_permille=2.0))
        else:
            L.append(UWLine("Diving", f"Max depth {depth} m (recreational)"))
    ms = av.get("motorsport", "none")
    if ms != "none":
        L.append(UWLine("Motorsport", ms.capitalize(), flat_extra_permille=3.0 if ms == "amateur" else 5.0))
    alt = av.get("mountaineering_m", 0)
    if alt > 8000:
        L.append(UWLine("Mountaineering", f"Expeditions above {alt} m", outcome="decline"))
    elif alt > 6000:
        L.append(UWLine("Mountaineering", f"Expeditions above {alt} m", flat_extra_permille=5.0))

    # Residence and travel (country of residence, never nationality)
    r = app.residence_risk_class
    if r == 4:
        L.append(UWLine("Residence", "Country risk class 4", outcome="decline"))
    else:
        L.append(UWLine("Residence", f"Country risk class {r}", flat_extra_permille={1: 0, 2: 1.0, 3: 3.0}[r]))
    t = app.travel_risk_class
    if t >= 3:
        L.append(UWLine("Travel", f"Frequent travel to risk class {t}",
                        flat_extra_permille=1.0 if t == 3 else 3.0, outcome="refer" if t == 4 else ""))

    _finalise(res, app)
    _evidence(res, app, pol)
    _financial(res, app, pol)
    return res


def _condition_lines(code: str, d: dict, age: int) -> list[UWLine]:
    label = CONDITIONS[code][0]
    if code == "diabetes_t2":
        h = d.get("hba1c", 7.0)
        if h >= 9:
            return [UWLine(label, f"HbA1c {h:.1f}%, poorly controlled", outcome="postpone")]
        debit = _band(h, [(7, 50), (8, 100), (9, 150)]) + (50 if age < 40 else 0)
        return [UWLine(label, f"HbA1c {h:.1f}%" + (", onset before 40" if age < 40 else ""), debit_pct=debit)]
    if code == "hypertension":
        return [UWLine(label, "Controlled on medication", debit_pct=15)]
    if code == "coronary":
        y = d.get("years_since_event", 0)
        if y < 1:
            return [UWLine(label, "Event within 12 months", outcome="postpone")]
        return [UWLine(label, f"{y} years since event", debit_pct=200 if age < 50 else 150)]
    if code == "cancer":
        typ, y = d.get("type", "other"), d.get("years_since_remission", 0)
        f = f"{typ.capitalize()}, {y} years since remission"
        rules = {
            "melanoma":   [(3, "postpone"), (5, (3.0, 3)), (99, None)],
            "breast":     [(2, "postpone"), (5, (5.0, 5)), (10, (2.0, 3)), (99, None)],
            "prostate":   [(1, "postpone"), (5, 50), (99, None)],
            "colorectal": [(3, "postpone"), (5, (7.5, 5)), (10, (2.5, 3)), (99, None)],
            "other":      [(5, "postpone"), (10, (5.0, 5)), (99, 25)],
        }[typ]
        r = _band(y, rules)
        if r == "postpone":
            return [UWLine(label, f, outcome="postpone")]
        if r is None:
            return [UWLine(label, f + ", standard")]
        if isinstance(r, tuple):
            return [UWLine(label, f, flat_extra_permille=r[0], flat_extra_years=r[1])]
        return [UWLine(label, f, debit_pct=r)]
    if code == "sleep_apnea":
        treated = d.get("treated", True)
        return [UWLine(label, "Treated" if treated else "Untreated", debit_pct=0 if treated else 50)]
    if code == "mental_health":
        s = d.get("severity", "mild")
        if s == "severe":
            return [UWLine(label, "Severe, history of hospitalisation", debit_pct=150, outcome="refer")]
        return [UWLine(label, s.capitalize(), debit_pct={"mild": 0, "moderate": 50}[s])]
    if code == "asthma":
        s = d.get("severity", "mild")
        return [UWLine(label, s.capitalize(), debit_pct={"mild": 0, "moderate": 25, "severe": 75}[s])]
    if code == "hepatitis":
        s = d.get("status", "cured")
        return [UWLine(label, s.capitalize(), debit_pct=25 if s == "cured" else 100)]
    if code == "alcohol":
        u = d.get("units_per_week", 0)
        if u > 50:
            return [UWLine(label, f"{u} units per week", outcome="decline")]
        return [UWLine(label, f"{u} units per week", debit_pct=_band(u, [(21, 0), (35, 25), (51, 75)]))]
    raise ValueError(code)


def _finalise(res: UWResult, app: Applicant) -> None:
    outcomes = {l.outcome for l in res.lines}
    debit = sum(l.debit_pct for l in res.lines)
    res.flat_extras = [(l.flat_extra_permille, l.flat_extra_years) for l in res.lines if l.flat_extra_permille > 0]

    # Preferred credits only for clean risks
    clean = (debit == 0 and not res.flat_extras and not app.conditions and not app.smoker
             and not app.family_history and outcomes <= {""})
    if clean and 18.5 <= app.bmi < 27 and app.systolic_bp < 130 and app.chol_hdl_ratio < 4.5:
        res.lines.append(UWLine("Preferred", "All preferred criteria met", debit_pct=-25))
        res.risk_class = "Preferred Plus"
    elif clean and app.bmi < 30 and app.systolic_bp < 140 and app.chol_hdl_ratio < 5.5:
        res.lines.append(UWLine("Preferred", "Preferred criteria met", debit_pct=-10))
        res.risk_class = "Preferred"

    res.total_debit = sum(l.debit_pct for l in res.lines)

    if "decline" in outcomes:
        res.decision, res.risk_class = "Decline", "Decline"
    elif "postpone" in outcomes:
        res.decision, res.risk_class = "Postpone", "Postpone"
    elif res.total_debit > MAX_DEBIT or res.max_flat_extra_permille > MAX_FLAT_EXTRA:
        res.decision, res.risk_class = "Decline", "Decline (rating above limits)"
    elif res.total_debit > 0 or res.flat_extras:
        res.decision = "Accept rated"
        if res.total_debit > 0:
            steps = int(np.ceil(res.total_debit / 25.0))
            res.total_debit = steps * 25.0           # round up to the next table
            res.risk_class = f"Table {TABLE_LETTERS[steps - 1]} (+{res.total_debit:.0f}%)"
        else:
            res.risk_class = "Standard with flat extra"
    if "refer" in outcomes and res.insurable:
        res.decision = "Accept rated"
        res.lines.append(UWLine("Referral", "Refer to reinsurer's chief underwriter", outcome="refer"))


# Evidence grid: (max sum assured, requirements, cost EUR)
EVIDENCE_GRID = [
    (1_000_000, ["Health questionnaire", "Blood profile"], 400),
    (5_000_000, ["Paramedical exam", "Blood profile", "Resting ECG", "Financial questionnaire"], 1_500),
    (10_000_000, ["Full medical exam (GP)", "Extended blood profile", "Resting ECG", "Financial questionnaire",
                  "Personal statement on financial need"], 3_500),
    (20_000_001, ["Full medical exam (specialist)", "Extended blood profile", "Exercise ECG",
                  "Cardiac echography", "Third-party financial report", "Inspection report"], 7_500),
]


def _evidence(res: UWResult, app: Applicant, pol: Policy) -> None:
    for limit, reqs, cost in EVIDENCE_GRID:
        if pol.sum_assured < limit:
            res.evidence, res.underwriting_cost = list(reqs), float(cost)
            break
    if app.age >= 65 and "Cognitive screening" not in res.evidence:
        res.evidence.append("Cognitive screening")
        res.underwriting_cost += 300
    if app.conditions:
        res.evidence.append("Attending physician statements")
        res.underwriting_cost += 250 * len(app.conditions)


def _financial(res: UWResult, app: Applicant, pol: Policy) -> None:
    sa = pol.sum_assured
    if app.purpose == "estate_planning":
        limit = 0.5 * app.net_worth
        basis = "50% of declared net worth (estate liquidity)"
    elif app.purpose == "key_person":
        limit = 10 * app.annual_income
        basis = "10x annual income (key person)"
    elif app.purpose == "loan_cover":
        limit = 0.8 * app.net_worth
        basis = "loan amount, assumed up to 80% of net worth"
    else:
        mult = 30 if app.age < 40 else 20 if app.age < 50 else 15 if app.age < 60 else 10
        limit = mult * app.annual_income
        basis = f"{mult}x annual income (family protection, age band)"
    if sa > limit:
        res.financial_flags.append(
            f"Sum assured EUR {sa:,.0f} exceeds the financial guideline of EUR {limit:,.0f} ({basis}). "
            "Justification and supporting documents required.")
    else:
        res.financial_flags.append(f"Within financial guideline of EUR {limit:,.0f} ({basis}).")
    if pol.term + app.age > 85:
        res.financial_flags.append("Policy expiry beyond age 85 is outside product limits.")
