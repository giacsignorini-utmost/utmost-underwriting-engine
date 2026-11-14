"""Jumbo Life Underwriting Engine: Streamlit front end."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (CONDITIONS, DEMO_CASES, Applicant, MedicalCondition, Policy, ReinsuranceProgram, Reinsurer,
                    demo_case, load_config, run_case, validate)
from engine.quote_pdf import build_quote_pdf

st.set_page_config(page_title="Jumbo Life Underwriting Engine", page_icon="◆", layout="wide")

NAVY, TEAL, GOLD, RED, GREY = "#0B2545", "#13807A", "#C49A2C", "#B5413B", "#8A94A6"
CFG = load_config()


def eur(x: float) -> str:
    return f"EUR {x:,.0f}"


# ----------------------------------------------------------------------------- state
def apply_case(app: Applicant, pol: Policy) -> None:
    s = st.session_state
    s.age, s.sex, s.height, s.weight = app.age, app.sex, app.height_cm, app.weight_kg
    s.bp, s.chol, s.smoker, s.family = app.systolic_bp, app.chol_hdl_ratio, app.smoker, app.family_history
    s.occ, s.res_class, s.travel_class = app.occupation_class, app.residence_risk_class, app.travel_risk_class
    av = app.avocations or {}
    s.pilot, s.dive = av.get("pilot_hours", 0), av.get("dive_depth_m", 0)
    s.motor, s.mount = av.get("motorsport", "none"), av.get("mountaineering_m", 0)
    s.income, s.networth, s.purpose = app.annual_income, app.net_worth, app.purpose
    s.conds = [CONDITIONS[c.code][0] for c in app.conditions]
    for c in app.conditions:
        for k, v in c.detail.items():
            s[f"cond_{c.code}_{k}"] = v
    s.sa, s.term, s.mode, s.unisex = pol.sum_assured, pol.term, pol.premium_mode, pol.unisex


def init_state() -> None:
    if "initialised" in st.session_state:
        return
    r, p = CFG["reinsurance"], CFG["pricing"]
    s = st.session_state
    s.retention, s.lines = r["retention"], r["surplus_lines"]
    s.treaty_rate, s.treaty_allow, s.brokerage = r["treaty_rate_pct"] * 100, r["treaty_first_year_allowance"] * 100, \
        r["brokerage"] * 100
    s.pool_base = pd.DataFrame(r["facultative_pool"]).assign(share=lambda d: d.share * 100, rate_pct=lambda d: d.rate_pct * 100)
    s.interest, s.margin = p["interest_rate"] * 100, p["profit_margin"] * 100
    s.comm1, s.comm_r = p["initial_commission"] * 100, p["renewal_commission"] * 100
    s.sigma, s.portfolio, s.nsims = CFG["simulation"]["mortality_level_sigma"] * 100, \
        CFG["simulation"]["portfolio_size"], CFG["simulation"]["n_sims"]
    s.demo = list(DEMO_CASES)[1]
    apply_case(*demo_case(s.demo))
    s.initialised = True


init_state()
S = st.session_state

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Case")
    demo = st.selectbox("Demo case", list(DEMO_CASES), key="demo")
    if st.button("Load demo case", width="stretch"):
        for k in [k for k in S.keys() if str(k).startswith("cond_")]:
            del S[k]
        apply_case(*demo_case(demo))
        st.rerun()

    with st.expander("Client", expanded=True):
        c1, c2 = st.columns(2)
        c1.number_input("Entry age", 18, 75, key="age")
        c2.selectbox("Sex", ["M", "F"], key="sex")
        c1.number_input("Height (cm)", 140.0, 215.0, step=1.0, key="height")
        c2.number_input("Weight (kg)", 40.0, 200.0, step=1.0, key="weight")
        c1.number_input("Systolic BP", 90, 220, key="bp")
        c2.number_input("Chol / HDL", 2.0, 12.0, step=0.1, key="chol")
        st.checkbox("Smoker", key="smoker")
        st.checkbox("Family history (CV or cancer death < 60)", key="family")

    with st.expander("Medical conditions", expanded=True):
        labels = {v[0]: k for k, v in CONDITIONS.items()}
        st.multiselect("Declared conditions", list(labels), key="conds")
        for lab in S.conds:
            code = labels[lab]
            for pk, spec in CONDITIONS[code][1].items():
                key = f"cond_{code}_{pk}"
                if isinstance(spec[1], list):
                    opts = spec[1]
                    if key not in S or S[key] not in opts:
                        S[key] = opts[0]
                    st.selectbox(f"{lab}: {spec[0]}", opts, key=key)
                else:
                    _, lo, hi, default, step = spec
                    if key not in S:
                        S[key] = default
                    st.number_input(f"{lab}: {spec[0]}", lo, hi, step=step, key=key)

    with st.expander("Lifestyle and residence"):
        st.selectbox("Occupation class", [1, 2, 3, 4], key="occ",
                     format_func=lambda x: {1: "1 Office or professional", 2: "2 Light manual",
                                            3: "3 Heavy manual or field travel", 4: "4 Hazardous"}[x])
        st.number_input("Private pilot hours per year", 0, 1000, key="pilot")
        st.number_input("Max diving depth (m)", 0, 150, key="dive")
        st.selectbox("Motorsport", ["none", "amateur", "professional"], key="motor")
        st.number_input("Mountaineering max altitude (m)", 0, 8849, step=100, key="mount")
        risk_lbl = {1: "1 Low", 2: "2 Moderate", 3: "3 Elevated", 4: "4 High"}
        st.selectbox("Country of residence risk class", [1, 2, 3, 4], key="res_class", format_func=risk_lbl.get)
        st.selectbox("Frequent travel risk class", [1, 2, 3, 4], key="travel_class", format_func=risk_lbl.get)
        st.caption("Country of residence and travel are rated. Nationality is never used.")

    with st.expander("Financial underwriting"):
        st.number_input("Annual income (EUR)", 0.0, 1e9, step=100_000.0, key="income", format="%.0f")
        st.number_input("Net worth (EUR)", 0.0, 1e10, step=1_000_000.0, key="networth", format="%.0f")
        st.selectbox("Purpose of cover", ["estate_planning", "key_person", "loan_cover", "family_protection"],
                     key="purpose", format_func=lambda x: x.replace("_", " ").capitalize())

    with st.expander("Product", expanded=True):
        st.number_input("Sum assured (EUR)", 100_000.0, 20_000_000.0, step=500_000.0, key="sa", format="%.0f")
        st.slider("Term (years)", 5, 30, key="term")
        st.radio("Premium", ["annual", "single"], key="mode", horizontal=True)
        st.checkbox("EU unisex pricing (Test-Achats)", key="unisex")

    with st.expander("Reinsurance programme"):
        st.number_input("Retention (EUR)", 100_000.0, 10_000_000.0, step=250_000.0, key="retention", format="%.0f")
        st.slider("Surplus treaty lines", 0, 20, key="lines")
        st.slider("Treaty YRT rate (% of best estimate)", 80.0, 150.0, step=1.0, key="treaty_rate")
        st.slider("Treaty first-year allowance (%)", 0.0, 100.0, step=5.0, key="treaty_allow")
        st.markdown("**Facultative pool** (via broker)")
        pool_df = st.data_editor(S.pool_base, key="pool_editor", num_rows="dynamic", width="stretch", hide_index=True,
                                column_config={"share": st.column_config.NumberColumn("Share %", format="%.0f"),
                                               "rate_pct": st.column_config.NumberColumn("Rate % BE", format="%.0f")})
        st.slider("Brokerage (% of fac premium)", 0.0, 25.0, step=0.5, key="brokerage")

    with st.expander("Pricing and simulation assumptions"):
        st.slider("Technical interest rate (%)", 0.0, 5.0, step=0.25, key="interest")
        st.slider("Target profit margin (% of premium)", 0.0, 25.0, step=1.0, key="margin")
        st.slider("Initial commission (% of first premium)", 0.0, 100.0, step=5.0, key="comm1")
        st.slider("Renewal commission (%)", 0.0, 15.0, step=0.5, key="comm_r")
        st.slider("Mortality level uncertainty (sigma %)", 0.0, 40.0, step=1.0, key="sigma")
        st.slider("Portfolio size (lives)", 1, 200, key="portfolio")
        st.select_slider("Simulations", [2_000, 5_000, 10_000, 20_000], key="nsims")

# ----------------------------------------------------------------------------- build inputs
labels = {v[0]: k for k, v in CONDITIONS.items()}
conditions = []
for lab in S.conds:
    code = labels[lab]
    conditions.append(MedicalCondition(code, {pk: S[f"cond_{code}_{pk}"] for pk in CONDITIONS[code][1]}))

app = Applicant(age=int(S.age), sex=S.sex, height_cm=S.height, weight_kg=S.weight, systolic_bp=int(S.bp),
                chol_hdl_ratio=S.chol, smoker=S.smoker, conditions=conditions, family_history=S.family,
                occupation_class=S.occ,
                avocations={"pilot_hours": S.pilot, "dive_depth_m": S.dive, "motorsport": S.motor,
                            "mountaineering_m": S.mount},
                residence_risk_class=S.res_class, travel_risk_class=S.travel_class, annual_income=S.income,
                net_worth=S.networth, purpose=S.purpose)
pol = Policy(sum_assured=float(S.sa), term=int(S.term), premium_mode=S.mode, unisex=S.unisex)

cfg = copy.deepcopy(CFG)
cfg["pricing"].update(interest_rate=S.interest / 100, profit_margin=S.margin / 100,
                      initial_commission=S.comm1 / 100, renewal_commission=S.comm_r / 100)
pool = pool_df.dropna()
ri = ReinsuranceProgram(
    retention=float(S.retention), surplus_lines=int(S.lines), treaty_rate_pct=S.treaty_rate / 100,
    treaty_first_year_allowance=S.treaty_allow / 100, treaty_renewal_allowance=0.0,
    facultative_pool=[Reinsurer(str(r["name"]), str(r["rating"]), float(r["share"]) / 100, float(r["rate_pct"]) / 100)
                      for _, r in pool.iterrows()],
    brokerage=S.brokerage / 100, concentration_alert=CFG["reinsurance"]["concentration_alert"])

# ----------------------------------------------------------------------------- header
st.markdown(f"<h1 style='color:{NAVY};margin-bottom:0'>Jumbo Life Underwriting Engine</h1>", unsafe_allow_html=True)
st.caption("Case study prepared for Utmost Group by Giacomo Signorini · Term life death cover up to EUR 20m · "
           "Underwriting, pricing, reinsurance placement and Solvency II capital in one flow · "
           "Illustrative assumptions, not an Utmost product")

errors = validate(app, pol, cfg)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

uw, model, res = run_case(app, pol, ri, cfg)

k = st.columns(5)
k[0].metric("Decision", uw.decision)
cls, _, extra = uw.risk_class.partition(" (")
k[1].metric("Risk class", cls, extra.rstrip(")") or None, delta_color="off", delta_arrow="off")
if res is None:
    st.warning(f"The case is not insurable as submitted: **{uw.decision}**. See the underwriting audit trail.")
else:
    unit = "/ year" if pol.premium_mode == "annual" else "single"
    k[2].metric(f"Gross premium {unit}", eur(res.premium), f"{res.rate_permille:.2f}‰ of SA", delta_color="off", delta_arrow="off")
    k[3].metric("PV profit net of reinsurance", eur(res.reinsurance_summary["PV profit net"]),
                f"gross {eur(res.reinsurance_summary['PV profit gross'])}", delta_color="off", delta_arrow="off")
    k[4].metric("SCR life relief", eur(res.capital["Capital relief"]),
                f"{res.capital['Capital relief'] / max(res.capital['SCR life gross'], 1):.0%} of gross SCR",
                delta_color="off", delta_arrow="off")
    for a in res.alerts:
        st.warning(a)

tabs = st.tabs(["Underwriting", "Pricing", "Reinsurance", "Risk and capital", "Cash flows", "Methodology"])

# ----------------------------------------------------------------------------- underwriting
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Audit trail")
        rows = []
        for l in uw.lines:
            fe = "" if not l.flat_extra_permille else (
                f"{l.flat_extra_permille:.1f}‰" + (f" for {l.flat_extra_years} yrs" if l.flat_extra_years else ""))
            rows.append({"Factor": l.factor, "Finding": l.finding,
                         "Debit": f"{l.debit_pct:+.0f}%" if l.debit_pct else "", "Flat extra": fe,
                         "Outcome": l.outcome.capitalize()})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(f"**Total mortality rating:** {uw.total_debit:+.0f}% · **Flat extras:** "
                    f"{uw.max_flat_extra_permille:.1f}‰ · **Class:** {uw.risk_class}")
    with c2:
        st.subheader("Evidence requirements")
        for e in uw.evidence:
            st.markdown(f"- {e}")
        st.caption(f"Estimated underwriting cost: {eur(uw.underwriting_cost)}")
        st.subheader("Financial underwriting")
        for f in uw.financial_flags:
            (st.info if f.startswith("Within") else st.warning)(f)
        st.metric("BMI", f"{app.bmi:.1f}")

if res is not None:
    # ------------------------------------------------------------------------- pricing
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Where the premium goes (PV)")
            d = res.decomposition
            parts = ["Expected claims", "Expenses and underwriting", "Distribution commission", "Profit margin"]
            fig = go.Figure(go.Bar(x=[d[p] for p in parts], y=parts, orientation="h",
                                   marker_color=[NAVY, GREY, GOLD, TEAL],
                                   text=[f"{eur(d[p])} ({d[p] / d['PV premiums']:.0%})" for p in parts],
                                   textposition="auto"))
            fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                              xaxis_title="EUR, present value")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Mortality by policy year")
            cf = res.cashflows
            fig = go.Figure()
            fig.add_scatter(x=cf["Policy year"], y=cf["q standard (per mille)"], name="Standard HNW basis",
                            line=dict(color=GREY, dash="dot"))
            fig.add_scatter(x=cf["Policy year"], y=cf["q rated (per mille)"], name="Rated (this life)",
                            line=dict(color=NAVY, width=3))
            fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=10), yaxis_title="q (per mille)",
                              xaxis_title="Policy year", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
        p_net = res.reinsurance_summary["Premium for target margin net of reinsurance"]
        st.info(f"The gross premium of **{eur(res.premium)}** meets the {S.margin:.0f}% margin before reinsurance. "
                f"To keep the same margin **after** the cost of this reinsurance programme, the premium would need "
                f"to be **{eur(p_net)}** ({p_net / res.premium - 1:+.1%}).")

    # ------------------------------------------------------------------------- reinsurance
    with tabs[2]:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Risk allocation")
            colors = {"Retention": NAVY, "Treaty": TEAL, "Facultative": GOLD}
            fig = go.Figure(go.Pie(labels=[l.name for l in res.layers], values=[l.sum_assured for l in res.layers],
                                   hole=0.55, marker=dict(colors=[colors[l.kind] for l in res.layers],
                                                          line=dict(color="white", width=2)),
                                   sort=False, textinfo="percent"))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=10),
                              annotations=[dict(text=eur(pol.sum_assured), showarrow=False, font_size=13)])
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Placement")
            pl = res.reinsurance_summary["Per layer PV premium"]
            st.dataframe(pd.DataFrame([{
                "Layer": l.name, "Type": l.kind, "Rating": l.rating, "Sum assured": eur(l.sum_assured),
                "Share": f"{l.sum_assured / pol.sum_assured:.1%}",
                "YRT rate": f"{l.rate_pct:.0%} of BE" if l.rate_pct else "",
                "PV reinsurance premium": eur(pl.get(l.name, 0.0))} for l in res.layers]),
                hide_index=True, width="stretch")
            r = res.reinsurance_summary
            st.dataframe(pd.DataFrame({"Item": ["PV reinsurance premiums", "less PV treaty allowances",
                                                "plus PV brokerage", "less PV expected recoveries",
                                                "Net cost of reinsurance"],
                                       "EUR": [eur(r["PV reinsurance premiums"]), eur(-r["PV treaty allowances"]),
                                               eur(r["PV brokerage"]), eur(-r["PV expected recoveries"]),
                                               eur(r["Net cost of reinsurance"])]}),
                         hide_index=True, width="stretch")
        st.subheader("Expected annual cash flows: gross vs ceded")
        cf = res.cashflows
        fig = go.Figure()
        fig.add_bar(x=cf["Policy year"], y=cf["Premium"], name="Gross premium", marker_color=NAVY)
        fig.add_bar(x=cf["Policy year"], y=-(cf["Reinsurance premium"] - cf["Allowance"] + cf["Brokerage"]),
                    name="Reinsurance cost", marker_color=GOLD)
        fig.add_scatter(x=cf["Policy year"], y=-cf["Expected claims"], name="Expected claims",
                        line=dict(color=RED))
        fig.add_scatter(x=cf["Policy year"], y=cf["Expected recoveries"], name="Expected recoveries",
                        line=dict(color=TEAL))
        fig.update_layout(barmode="relative", height=340, margin=dict(l=0, r=10, t=10, b=10),
                          yaxis_title="EUR (expected, undiscounted)", xaxis_title="Policy year",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------------- risk & capital
    with tabs[3]:
        st.subheader("Single policy: exact distribution of PV profit")
        st.caption("On one life the result has only 2n + 1 possible outcomes, so the distribution is computed "
                   "exactly rather than simulated.")
        out = res.outcomes
        deaths = out[out["Outcome"].str.startswith("Death")]
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            fig.add_bar(x=deaths["Outcome"].str.replace("Death in year ", ""), y=deaths["PV profit gross"],
                        name="Gross", marker_color=RED)
            fig.add_bar(x=deaths["Outcome"].str.replace("Death in year ", ""), y=deaths["PV profit net"],
                        name="Net of reinsurance", marker_color=TEAL)
            fig.update_layout(barmode="group", height=320, margin=dict(l=0, r=10, t=10, b=10),
                              xaxis_title="Year of death", yaxis_title="PV profit (EUR)",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
        with c2:
            g, n = res.exact_stats["gross"], res.exact_stats["net"]
            st.dataframe(pd.DataFrame({
                "Metric": ["Expected PV profit", "Standard deviation", "Probability of loss", "VaR 99.5% (loss)",
                           "Worst outcome"],
                "Gross": [eur(g["mean"]), eur(g["std"]), f"{g['prob_loss']:.2%}", eur(g["VaR 99.5% loss"]),
                          eur(g["worst"])],
                "Net": [eur(n["mean"]), eur(n["std"]), f"{n['prob_loss']:.2%}", eur(n["VaR 99.5% loss"]),
                        eur(n["worst"])]}), hide_index=True, width="stretch")

        st.subheader(f"Monte Carlo: portfolio of {S.portfolio} similar lives")
        st.caption(f"{S.nsims:,} scenarios. Each scenario draws a common mortality level (lognormal, sigma "
                   f"{S.sigma:.0f}%), the systematic risk that does not diversify, then simulates every life.")

        @st.cache_data(show_spinner="Simulating scenarios...")
        def simulate(_model, key, premium, nsims, portfolio, sigma):
            return _model.simulate(premium, nsims, portfolio, sigma, CFG["simulation"]["seed"])

        key = (repr(app), repr(pol), repr(ri), repr(cfg["pricing"]))
        sim = simulate(model, key, res.premium, int(S.nsims), int(S.portfolio), S.sigma / 100)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            fig.add_histogram(x=sim["gross"], name="Gross", marker_color=RED, opacity=0.55, nbinsx=80)
            fig.add_histogram(x=sim["net"], name="Net of reinsurance", marker_color=TEAL, opacity=0.75, nbinsx=80)
            for st_, colr in ((sim["stats_gross"], RED), (sim["stats_net"], TEAL)):
                fig.add_vline(x=-st_["VaR 99.5% loss"], line=dict(color=colr, dash="dash"))
            fig.update_layout(barmode="overlay", height=340, margin=dict(l=0, r=10, t=10, b=10),
                              xaxis_title="Portfolio PV profit (EUR) · dashed lines: 99.5% VaR",
                              yaxis_title="Scenarios", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
        with c2:
            g, n = sim["stats_gross"], sim["stats_net"]
            st.dataframe(pd.DataFrame({
                "Metric": ["Expected PV profit", "Standard deviation", "Probability of loss", "VaR 99.5% (loss)",
                           "TVaR 99.5% (loss)"],
                "Gross": [eur(g["mean"]), eur(g["std"]), f"{g['prob_loss']:.1%}", eur(g["VaR 99.5% loss"]),
                          eur(g["TVaR 99.5% loss"])],
                "Net": [eur(n["mean"]), eur(n["std"]), f"{n['prob_loss']:.1%}", eur(n["VaR 99.5% loss"]),
                        eur(n["TVaR 99.5% loss"])]}), hide_index=True, width="stretch")
            st.caption(f"Average claims per scenario: {sim['claims'].mean():.2f}")

        st.subheader("Solvency II life underwriting risk (simplified standard formula)")
        c = res.capital
        c1, c2 = st.columns([3, 2])
        with c1:
            cats = ["Mortality (+15%)", "Catastrophe (+1.5‰)", "Life SCR (ρ = 0.25)"]
            fig = go.Figure()
            fig.add_bar(x=cats, y=[c["SCR mortality gross"], c["SCR catastrophe gross"], c["SCR life gross"]],
                        name="Gross", marker_color=NAVY)
            fig.add_bar(x=cats, y=[c["SCR mortality net"], c["SCR catastrophe net"], c["SCR life net"]],
                        name="Net", marker_color=TEAL)
            fig.update_layout(barmode="group", height=320, margin=dict(l=0, r=10, t=10, b=10), yaxis_title="EUR",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
        with c2:
            cost = res.reinsurance_summary["Net cost of reinsurance"]
            saved = c["PV cost of capital saved"]
            st.metric("SCR released by reinsurance", eur(c["Capital relief"]))
            st.metric(f"PV cost of capital saved ({CFG['capital']['cost_of_capital']:.0%} hurdle)", eur(saved))
            st.metric("PV net cost of reinsurance", eur(cost))
            verdict = "pays for itself in capital terms" if saved >= cost else "costs more than the capital it saves"
            st.info(f"On these assumptions the reinsurance programme **{verdict}**. The decision also reflects "
                    "risk appetite: the single policy worst case falls from "
                    f"{eur(-res.exact_stats['gross']['worst'])} to {eur(-res.exact_stats['net']['worst'])}.")

    # ------------------------------------------------------------------------- cash flows
    with tabs[4]:
        st.subheader("Expected cash flows by policy year")
        st.dataframe(res.cashflows.style.format({c: "{:,.0f}" for c in res.cashflows.columns
                                                 if c not in ("Policy year", "Attained age", "In force",
                                                              "q standard (per mille)", "q rated (per mille)")}
                                                | {"In force": "{:.4f}", "q standard (per mille)": "{:.3f}",
                                                   "q rated (per mille)": "{:.3f}"}),
                     width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download cash flows (CSV)", res.cashflows.to_csv(index=False).encode(),
                           "cashflows.csv", "text/csv", width="stretch")
        c2.download_button("Download quote sheet (PDF)", build_quote_pdf(app, pol, uw, res), "quote_sheet.pdf",
                           "application/pdf", width="stretch", type="primary")

with tabs[5]:
    st.markdown((Path(__file__).parent / "METHODOLOGY.md").read_text(encoding="utf-8"))
