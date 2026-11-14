# Life Underwriting Engine

**An end-to-end pricing lab for high sum assured term life cover: from the medical file to the capital at risk net of reinsurance.**

Case study prepared for the Product Team of Utmost Group by **Giacomo Signorini**. All assumptions are illustrative.

## What it does

A wealthy client asks for up to EUR 20m of death cover. The engine takes the case through the same steps an insurer would, in one flow:

| Step | Input | Output |
|---|---|---|
| **Underwriting** | Biometrics, medical conditions, occupation, avocations, country of residence and travel, finances | Risk class, debits and flat extras, evidence required, financial underwriting alerts, full audit trail |
| **Pricing** | Product, term, premium mode, pricing assumptions | Gross premium, rate per mille, decomposition of the premium |
| **Reinsurance** | Retention, surplus treaty, facultative pool via broker, brokerage | Placement by layer and reinsurer, net cost of reinsurance, premium needed to keep the margin after reinsurance |
| **Risk** | Portfolio size, mortality level uncertainty | Exact profit distribution of the single policy, Monte Carlo on a portfolio, VaR and TVaR at 99.5%, gross vs net |
| **Capital** | Solvency II shocks, cost of capital | SCR mortality and catastrophe, gross vs net, capital relief vs cost of reinsurance |

A one-page **quote sheet in PDF** and the expected cash flows in CSV can be downloaded from the app.

## Design choices worth noting

* **One outcome model behind every number.** A policy on one life ends in one of 2n + 1 ways (death in year t, lapse after year t, survival). The premium, expected profit, exact distribution, Monte Carlo and capital shocks all come from this single model, so they reconcile by construction. The tests check that they do.
* **Exact where possible, simulated where necessary.** On one life the distribution is computed exactly. Monte Carlo is used where it adds information: a portfolio of lives with a common, non-diversifiable uncertainty on the mortality level.
* **Closed-form premium.** PV profit is linear in the premium, so the target margin is met exactly in two evaluations, with no iteration.
* **Compliant by design.** EU unisex pricing (Test-Achats) is the default. Country of residence and travel are rated; nationality is never used.
* **Reinsurance judged on capital, not only on cost.** The app compares the PV net cost of reinsurance with the PV cost of the capital it releases.

The full methodology, with formulas and declared limits, is in [METHODOLOGY.md](METHODOLOGY.md) and in the app's Methodology tab.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the tests:

```bash
pip install pytest
pytest
```

## Structure

```
engine/
  models.py        Data structures (applicant, policy, reinsurance programme)
  mortality.py     Gompertz-Makeham basis, select, smoker and HNW factors
  underwriting.py  Numerical rating engine, evidence grid, financial underwriting
  case.py          Outcome model: pricing, reinsurance, exact distribution, Monte Carlo, capital
  quote_pdf.py     One-page quote sheet
config/
  assumptions.yaml Every assumption, editable without touching the code
streamlit_app.py   Web interface
tests/             Automated tests (pytest)
```

## Disclaimer

Ratings, mortality parameters, commissions and reinsurance terms are assumptions chosen by the author for illustration. They do not reflect the pricing or underwriting basis of any insurer or reinsurer, including the author's current or former employers. Not an offer of insurance.
