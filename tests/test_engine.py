import copy

import numpy as np
import pytest

from engine import (Applicant, MedicalCondition, Policy, DEMO_CASES, assess, demo_case, load_config,
                    program_from_config, run_case, validate)
from engine.case import CaseModel, outcome_probs

CFG = load_config()
RI = program_from_config(CFG)


def _run(app, pol, cfg=CFG, ri=RI):
    return run_case(app, pol, ri, cfg)


@pytest.mark.parametrize("name", list(DEMO_CASES))
def test_demo_cases_run(name):
    app, pol = demo_case(name)
    assert not validate(app, pol, CFG)
    uw, model, res = _run(app, pol)
    assert uw.insurable and res.premium > 0


@pytest.mark.parametrize("name", list(DEMO_CASES))
def test_outcome_probabilities_sum_to_one(name):
    app, pol = demo_case(name)
    _, model, _ = _run(app, pol)
    assert outcome_probs(model.q, model.w).sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("mode", ["annual", "single"])
def test_premium_hits_target_margin(mode):
    app, pol = demo_case(list(DEMO_CASES)[1])
    pol.premium_mode = mode
    _, _, res = _run(app, pol)
    d = res.decomposition
    assert d["Profit margin"] == pytest.approx(CFG["pricing"]["profit_margin"] * d["PV premiums"], rel=1e-9)
    total = d["Expected claims"] + d["Expenses and underwriting"] + d["Distribution commission"] + d["Profit margin"]
    assert total == pytest.approx(d["PV premiums"], rel=1e-9)


def test_net_premium_hits_target_margin_after_reinsurance():
    app, pol = demo_case(list(DEMO_CASES)[2])
    _, model, res = _run(app, pol)
    p_net = res.reinsurance_summary["Premium for target margin net of reinsurance"]
    probs = outcome_probs(model.q, model.w)
    _, net = model.outcome_values(p_net)
    l = np.concatenate([[1.0], np.cumprod((1 - model.q) * (1 - model.w))])[:-1]
    pv_prem = p_net * float(l @ model.v ** model.t)
    assert probs @ net == pytest.approx(CFG["pricing"]["profit_margin"] * pv_prem, rel=1e-9)


@pytest.mark.parametrize("sa", [500_000, 1_000_000, 7_000_000, 10_000_000, 15_500_000, 20_000_000])
def test_layers_sum_to_sum_assured(sa):
    app, pol = demo_case(list(DEMO_CASES)[0])
    pol.sum_assured = sa
    _, _, res = _run(app, pol)
    assert sum(l.sum_assured for l in res.layers) == pytest.approx(sa)
    assert res.layers[0].sum_assured <= RI.retention + 1e-6


def test_premium_increases_with_rating():
    app, pol = demo_case(list(DEMO_CASES)[0])
    base = _run(app, pol)[2].premium
    app.conditions = [MedicalCondition("diabetes_t2", {"hba1c": 7.5})]
    rated = _run(app, pol)[2].premium
    app.avocations = {"pilot_hours": 20}
    rated_fe = _run(app, pol)[2].premium
    assert base < rated < rated_fe


def test_smoker_costs_more():
    app, pol = demo_case(list(DEMO_CASES)[0])
    p_ns = _run(app, pol)[2].premium
    app.smoker = True
    assert _run(app, pol)[2].premium > p_ns


def test_unisex_pricing_ignores_sex():
    app, pol = demo_case(list(DEMO_CASES)[0])
    m = _run(app, pol)[2].premium
    app.sex = "F"
    assert _run(app, pol)[2].premium == pytest.approx(m)
    pol.unisex = False
    assert _run(app, pol)[2].premium < m


def test_knockouts():
    app, pol = demo_case(list(DEMO_CASES)[0])
    a = copy.deepcopy(app); a.residence_risk_class = 4
    assert assess(a, pol).decision == "Decline"
    a = copy.deepcopy(app); a.systolic_bp = 190
    assert assess(a, pol).decision == "Postpone"
    a = copy.deepcopy(app); a.conditions = [MedicalCondition("coronary", {"years_since_event": 0})]
    assert assess(a, pol).decision == "Postpone"
    a = copy.deepcopy(app); a.conditions = [MedicalCondition("alcohol", {"units_per_week": 60})]
    assert assess(a, pol).decision == "Decline"


def test_table_rating_rounds_up_to_25():
    app, pol = demo_case(list(DEMO_CASES)[0])
    app.conditions = [MedicalCondition("hypertension")]   # +15% -> Table A (+25%)
    uw = assess(app, pol)
    assert uw.total_debit == 25 and uw.risk_class.startswith("Table A")


def test_reinsurance_reduces_capital():
    for name in DEMO_CASES:
        app, pol = demo_case(name)
        cap = _run(app, pol)[2].capital
        assert cap["SCR life net"] < cap["SCR life gross"]


def test_monte_carlo_matches_exact_mean():
    app, pol = demo_case(list(DEMO_CASES)[1])
    _, model, res = _run(app, pol)
    sim = model.simulate(res.premium, n_sims=20_000, portfolio=20, sigma=0.0, seed=7)
    exact = 20 * res.exact_stats["net"]["mean"]
    se = sim["net"].std() / np.sqrt(len(sim["net"]))
    assert abs(sim["net"].mean() - exact) < 4 * se


def test_exact_var_single_policy():
    app, pol = demo_case(list(DEMO_CASES)[1])
    _, _, res = _run(app, pol)
    g = res.exact_stats["gross"]
    assert g["VaR 99.5% loss"] <= -g["worst"] + 1e-6
    assert g["TVaR 99.5% loss"] >= g["VaR 99.5% loss"] - 1e-6


def test_validation():
    app = Applicant(age=80)
    assert validate(app, Policy(sum_assured=25_000_000, term=10), CFG)
