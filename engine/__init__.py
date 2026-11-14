"""Utmost Underwriting Engine: illustrative jumbo life underwriting, pricing and reinsurance."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .case import CaseModel, CaseResult
from .models import Applicant, MedicalCondition, Policy, ReinsuranceProgram, Reinsurer
from .underwriting import CONDITIONS, UWResult, assess

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "assumptions.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def program_from_config(cfg: dict) -> ReinsuranceProgram:
    r = cfg["reinsurance"]
    return ReinsuranceProgram(
        retention=r["retention"], surplus_lines=r["surplus_lines"], treaty_rate_pct=r["treaty_rate_pct"],
        treaty_first_year_allowance=r["treaty_first_year_allowance"],
        treaty_renewal_allowance=r["treaty_renewal_allowance"],
        facultative_pool=[Reinsurer(**x) for x in r["facultative_pool"]],
        brokerage=r["brokerage"], concentration_alert=r["concentration_alert"])


def validate(app: Applicant, pol: Policy, cfg: dict) -> list[str]:
    p = cfg["product"]
    errors = []
    if not p["min_entry_age"] <= app.age <= p["max_entry_age"]:
        errors.append(f"Entry age must be between {p['min_entry_age']} and {p['max_entry_age']}.")
    if not p["min_term"] <= pol.term <= p["max_term"]:
        errors.append(f"Term must be between {p['min_term']} and {p['max_term']} years.")
    if app.age + pol.term > p["max_expiry_age"]:
        errors.append(f"Expiry age {app.age + pol.term} exceeds the maximum of {p['max_expiry_age']}.")
    if not 0 < pol.sum_assured <= p["max_sum_assured"]:
        errors.append(f"Sum assured must be positive and at most EUR {p['max_sum_assured']:,.0f}.")
    return errors


def run_case(app: Applicant, pol: Policy, ri: ReinsuranceProgram, cfg: dict):
    """Underwrite and, if insurable, price the case. Returns (UWResult, CaseModel | None, CaseResult | None)."""
    uw = assess(app, pol)
    if not uw.insurable:
        return uw, None, None
    model = CaseModel(app, pol, uw, ri, cfg)
    return uw, model, model.run()


DEMO_CASES = {
    "Healthy entrepreneur, 45, EUR 5m": dict(
        applicant=Applicant(age=45, sex="M", height_cm=182, weight_kg=79, systolic_bp=124, chol_hdl_ratio=3.8,
                            annual_income=1_200_000, net_worth=25_000_000, purpose="estate_planning"),
        policy=Policy(sum_assured=5_000_000, term=20)),
    "Executive, 58, diabetic private pilot, EUR 12m": dict(
        applicant=Applicant(age=58, sex="M", height_cm=176, weight_kg=92, systolic_bp=138, chol_hdl_ratio=4.9,
                            conditions=[MedicalCondition("diabetes_t2", {"hba1c": 7.4}),
                                        MedicalCondition("hypertension")],
                            avocations={"pilot_hours": 80}, annual_income=2_500_000, net_worth=40_000_000,
                            purpose="estate_planning"),
        policy=Policy(sum_assured=12_000_000, term=15)),
    "Founder, 52, EUR 20m, risk class 3 residence": dict(
        applicant=Applicant(age=52, sex="F", height_cm=168, weight_kg=63, systolic_bp=128, chol_hdl_ratio=4.2,
                            conditions=[MedicalCondition("cancer", {"type": "breast", "years_since_remission": 7})],
                            residence_risk_class=3, travel_risk_class=3, annual_income=3_000_000,
                            net_worth=60_000_000, purpose="key_person"),
        policy=Policy(sum_assured=20_000_000, term=10)),
}


def demo_case(name: str):
    d = DEMO_CASES[name]
    return copy.deepcopy(d["applicant"]), copy.deepcopy(d["policy"])


__all__ = ["Applicant", "MedicalCondition", "Policy", "ReinsuranceProgram", "Reinsurer", "CONDITIONS",
           "UWResult", "CaseModel", "CaseResult", "assess", "load_config", "program_from_config", "run_case",
           "validate", "DEMO_CASES", "demo_case"]
