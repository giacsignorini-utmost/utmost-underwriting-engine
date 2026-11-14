## Methodology

The engine follows one case from the application to the capital position net of reinsurance. Every assumption lives in `config/assumptions.yaml` and is illustrative: none is taken from an insurer's or reinsurer's rating manual or pricing basis.

### 1. Mortality basis

Base mortality follows a Gompertz-Makeham law, $\mu(x) = A + B c^x$, which gives the one-year death probability

$$q_x = 1 - \exp\left(-A - B c^x \frac{c-1}{\ln c}\right).$$

The standard basis for a policy year $t$ applies, in this order: select factors for the first five years after underwriting, a high-net-worth factor (insured lives in the top income bands have lighter mortality) and a smoker factor where relevant.

**EU unisex pricing.** Since the Test-Achats judgment (CJEU C-236/09, 2011), sex may not lead to different premiums in the EU. By default the engine prices on a fixed gender mix (70% male). Switching the option off shows what the sex-specific price would have been.

### 2. Underwriting

A numerical rating system, as used in life underwriting manuals:

* **Percentage debits and credits** on mortality for build, blood pressure, lipids, family history and medical conditions
* **Flat extras** in per mille of the sum assured, for the whole term or a limited period, for occupation, aviation, diving, motorsport, mountaineering, country of residence and travel, and some cancer histories
* **Preferred credits** (−10% or −25%) for clean risks only
* **Postpone or decline** for knock-out findings, or when the total rating goes above +300% or 10‰

The total debit is rounded up to the next table (Table A = +25%, Table B = +50%, and so on). The rated mortality is

$$q'_t = q^{std}_t \,(1 + d) + f_t.$$

**Country of residence, not nationality.** The engine rates where the client lives and travels, which drives the actual exposure. Nationality is never used: it is not a mortality driver in itself and pricing on it would create discrimination risk.

**Evidence and financial underwriting.** The sum assured sets the medical and financial evidence required and its cost, which goes into the initial expenses. Financial underwriting checks the sum assured against guidelines that depend on the purpose of the cover (estate liquidity, key person, loan cover, family protection). It raises alerts and does not change the price.

### 3. The outcome model

A single policy ends in exactly one of $2n + 1$ ways: death in year $t$, lapse at the end of year $t$, or survival to expiry. With $l_t$ the probability of being in force at the start of year $t$ and $w_t$ the lapse rate:

$$P(\text{death in } t) = l_t\, q'_t, \qquad P(\text{lapse after } t) = l_t (1 - q'_t)\, w_t, \qquad P(\text{survival}) = l_n.$$

Each outcome has a present value of all cash flows, both gross and net of reinsurance. Premiums, commissions, expenses and reinsurance premiums are paid at the start of each year; claims and recoveries at the end of the year of death. **Every result in the app comes from this one model**: the premium, the expected profit, the exact distribution, the Monte Carlo simulation and the capital shocks. They are therefore consistent with each other by construction.

### 4. Pricing

The premium $P$ solves

$$E[\text{PV profit}] = m \cdot \text{PV}(\text{premiums}),$$

where $m$ is the target margin. Because PV profit is linear in $P$, two evaluations give the exact solution, with no iteration. The app also shows the premium needed to hit the same margin **after** the net cost of reinsurance, a figure that is often left out of jumbo case pricing.

### 5. Reinsurance

* **Retention**: the company keeps up to its retention limit.
* **Surplus treaty**: automatic capacity of a number of lines times the retention.
* **Facultative**: the excess is placed through a broker with a pool of reinsurers, each with its own share, rating and rate.

Reinsurance is on a **yearly renewable term (YRT)** basis. Each reinsurer charges a percentage of the company's best estimate mortality, with the same debits, and flat extras passed on at 100%. The treaty may return a first-year allowance; the cedant pays brokerage on the facultative premium. The net cost of reinsurance is the PV of reinsurance premiums, less allowances, plus brokerage, less expected recoveries.

### 6. Risk

**Exact distribution.** On one life, simulation adds nothing: the $2n + 1$ outcomes and their probabilities give the full distribution. The app reports the mean, standard deviation, probability of loss, 99.5% VaR and the worst case, gross and net.

**Monte Carlo.** Simulation matters for a portfolio and for uncertainty on the parameters. Each scenario draws a common mortality level $\theta \sim \text{Lognormal}(-\sigma^2/2, \sigma)$, so $E[\theta] = 1$. This is the systematic risk that does not diversify away. Each life in the portfolio then ends in one of its outcomes with probabilities computed from $\theta q'$. With $\sigma = 0$ the simulated mean converges to the exact mean, and the test suite checks this.

### 7. Capital

A simplified Solvency II standard formula for life underwriting risk:

* **Mortality**: the loss of value under a permanent +15% increase in mortality
* **Catastrophe**: +1.5‰ absolute mortality over the next 12 months on the capital at risk. Unlike the mortality shock, it does not depend on the health of the life: it scales directly with the sum assured, so on a jumbo case only reinsurance can bring it down.
* Aggregated with a correlation of 0.25

Capital is projected to run off with the in-force and, for mortality, with the remaining term. Its PV cost at the hurdle rate is compared with the net cost of reinsurance. The question it answers: does the reinsurance pay for itself in capital terms?

### Limits, declared

* Ratings, mortality parameters, commissions and reinsurance terms are illustrative.
* The capital calculation covers mortality and catastrophe risk only. It leaves out lapse and expense risk, counterparty default risk on reinsurers, diversification with the rest of the balance sheet, and the risk margin.
* Reinsurance rates are assumed guaranteed for the term. In practice YRT rates are often reviewable.
* Discounting uses a flat rate rather than the EIOPA risk-free curve.
