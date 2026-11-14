"""Data structures shared across the engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MedicalCondition:
    """One declared condition. `detail` carries the condition specific parameters."""
    code: str
    detail: dict = field(default_factory=dict)


@dataclass
class Applicant:
    age: int
    sex: str = "M"                      # "M" or "F"; priced unisex unless switched off
    height_cm: float = 180.0
    weight_kg: float = 78.0
    systolic_bp: int = 125
    chol_hdl_ratio: float = 4.0
    smoker: bool = False
    conditions: list[MedicalCondition] = field(default_factory=list)
    family_history: bool = False         # parent or sibling died of CV disease or cancer before 60
    occupation_class: int = 1            # 1..4
    avocations: dict = field(default_factory=dict)
    # avocations keys: pilot_hours (int), dive_depth_m (int), motorsport ("none","amateur","professional"),
    #                  mountaineering_m (int)
    residence_risk_class: int = 1        # 1..4, country of residence (never nationality)
    travel_risk_class: int = 1           # 1..4, highest risk class of frequent travel
    annual_income: float = 500_000.0
    net_worth: float = 10_000_000.0
    purpose: str = "estate_planning"     # estate_planning, key_person, loan_cover, family_protection

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_cm / 100.0) ** 2


@dataclass
class Policy:
    sum_assured: float
    term: int
    premium_mode: str = "annual"         # "annual" or "single"
    unisex: bool = True


@dataclass
class Reinsurer:
    name: str
    rating: str
    share: float
    rate_pct: float


@dataclass
class ReinsuranceProgram:
    retention: float
    surplus_lines: int
    treaty_rate_pct: float
    treaty_first_year_allowance: float
    treaty_renewal_allowance: float
    facultative_pool: list[Reinsurer]
    brokerage: float
    concentration_alert: float = 0.5


@dataclass
class UWLine:
    factor: str
    finding: str
    debit_pct: float = 0.0
    flat_extra_permille: float = 0.0
    flat_extra_years: Optional[int] = None   # None means whole term
    outcome: str = ""                        # "", "postpone", "decline", "refer"
