"""Post-deployment drift scenarios and realistic, decision-dependent label delays.

Every scenario replays the *same* baseline world: the injected incident draws from the
random generator only after all baseline events exist, so events before the change day are
identical across scenarios and the ``none`` scenario is an exact counterfactual.

Scenarios (all synthetic, chosen to stress different monitors):

* ``none``: nothing changes. Any alarm here is a false alarm.
* ``ato_wave``: account takeovers quadruple and most come from the victim's own device and
  network (session hijacking). The model sees little; losses arrive as late chargebacks.
* ``evasive_bots``: a new, unseen traffic source sends bot sign-ups that mimic human
  click-to-conversion time and rotate devices.
* ``partner_campaign``: a legitimate partner launches e-mail blasts that bring bursts of
  genuine new customers. Nothing is fraudulent, but the bursts look like bot traffic to the
  model, so good customers get declined, and declined events never receive a label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rde.config import SimulationConfig
from rde.data.synthetic import COUNTRIES, COUNTRY_WEIGHTS, DAY, EVENT_COLUMNS, _Sim

SCENARIOS = ("none", "ato_wave", "evasive_bots", "partner_campaign")


@dataclass(frozen=True)
class ScenarioConfig:
    change_day: int = 120
    ato_multiplier: float = 3.0  # extra ATO incidents, as a multiple of the baseline rate
    ato_hijack_share: float = 0.7
    bot_multiplier: float = 2.0
    partner_blasts_per_day: float = 1.0
    partner_signups_per_blast: tuple[int, int] = (15, 45)


class _ScenarioSim(_Sim):
    def __init__(self, cfg: SimulationConfig, scenario: str, sc: ScenarioConfig):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        super().__init__(cfg)
        self.scenario, self.sc = scenario, sc
        self.t_change = self.t0 + sc.change_day * DAY
        self.days_after = max(self.t_end - self.t_change, 0.0) / DAY
        self.scale_after = self.days_after * cfg.n_users / 10_000

    def _ato_wave(self, users: list[dict]) -> None:
        cfg, rng, sc = self.cfg, self.rng, self.sc
        established = [u for u in users if u["created"] < self.t0]
        n = rng.poisson(cfg.ato_incidents_per_10k_users_day * sc.ato_multiplier * self.scale_after)
        for _ in range(n):
            u = established[rng.integers(len(established))]
            ts = rng.uniform(self.t_change, self.t_end)
            if rng.random() < sc.ato_hijack_share:
                device, ip, ip_cc = rng.choice(u["devices"]), rng.choice(u["ips"]), u["country"]
            else:
                device, ip, ip_cc = self._new("d"), self._new("ip"), rng.choice(COUNTRIES)
            for _ in range(rng.integers(1, 4)):
                ts += rng.exponential(1800)
                amount = round(float(rng.lognormal(np.log(180), 0.6)), 2)
                self._emit(
                    ts,
                    "purchase",
                    u["id"],
                    device,
                    ip,
                    ip_cc,
                    u["country"],
                    "organic",
                    u["pm"],
                    amount,
                    u["created"],
                    None,
                    1,
                    "account_takeover",
                )

    def _evasive_bots(self) -> None:
        cfg, rng, sc = self.cfg, self.rng, self.sc
        n = rng.poisson(cfg.bot_campaigns_per_10k_users_day * sc.bot_multiplier * self.scale_after)
        for _ in range(n):
            start = rng.uniform(self.t_change, self.t_end)
            duration = rng.uniform(4, 12) * 3600  # slower drip than the known farms
            ips = [self._new("ip") for _ in range(rng.integers(10, 30))]
            cc = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
            for _ in range(rng.integers(20, 80)):
                ts = start + rng.uniform(0, duration)
                self._emit(
                    ts,
                    "signup",
                    self._new("f"),
                    self._new("d"),
                    rng.choice(ips),
                    cc,
                    cc,
                    "src_new_affiliate",
                    None,
                    0.0,
                    ts,
                    self._ctc_human(),
                    1,
                    "bot_farm",
                )

    def _partner_campaign(self) -> None:
        rng, sc = self.rng, self.sc
        for _ in range(rng.poisson(sc.partner_blasts_per_day * self.days_after)):
            start = rng.uniform(self.t_change, self.t_end)
            lo, hi = sc.partner_signups_per_blast
            for _ in range(rng.integers(lo, hi + 1)):
                created = start + rng.exponential(1200)  # most clicks in the first half hour
                user, cc = (
                    self._new("f").replace("fu_", "pu_"),
                    rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS),
                )
                device, ip, pm = self._new("d"), self._new("ip"), rng.choice(["card", "wallet"])
                self._emit(
                    created,
                    "signup",
                    user,
                    device,
                    ip,
                    cc,
                    cc,
                    "src_partner",
                    None,
                    0.0,
                    created,
                    float(rng.lognormal(np.log(120), 0.8)),
                    0,
                    "legit",
                )
                # Welcome-offer purchases: genuine, small, soon after sign-up.
                ts = created
                for _ in range(rng.integers(0, 3)):
                    ts += rng.exponential(2 * DAY)
                    amount = round(float(rng.lognormal(np.log(35), 0.5)), 2)
                    self._emit(
                        ts,
                        "purchase",
                        user,
                        device,
                        ip,
                        cc,
                        cc,
                        "organic",
                        pm,
                        amount,
                        created,
                        None,
                        0,
                        "legit",
                    )

    def run(self) -> pd.DataFrame:
        users = self.legit()
        self.card_testing()
        self.account_takeover(users)
        self.bot_farms()
        if self.scenario == "ato_wave":
            self._ato_wave(users)
        elif self.scenario == "evasive_bots":
            self._evasive_bots()
        elif self.scenario == "partner_campaign":
            self._partner_campaign()
        df = pd.DataFrame(self.rows, columns=EVENT_COLUMNS[1:])
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        df.insert(0, "event_id", [f"evt_{i:08d}" for i in range(len(df))])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["account_created_at"] = pd.to_datetime(df["account_created_at"], unit="s", utc=True)
        df["injected"] = df["timestamp"] >= pd.Timestamp(self.t_change, unit="s", tz="UTC")
        return df[[*EVENT_COLUMNS, "injected"]]


def generate_scenario(
    cfg: SimulationConfig, scenario: str, sc: ScenarioConfig | None = None
) -> pd.DataFrame:
    """Events for one scenario. The ``injected`` column only marks the post-change period."""
    return _ScenarioSim(cfg, scenario, sc or ScenarioConfig()).run()


# -- label delays -----------------------------------------------------------------------

#: Median days from event to a confirmed fraud label, for approved fraud, by pattern.
#: Illustrative assumptions: victims report takeovers quickly, chargebacks take weeks,
#: advertisers reconcile fake installs monthly, friendly fraud disputes come last.
LABEL_DELAY_MEDIAN_DAYS = {
    "account_takeover": 6.0,
    "card_testing": 20.0,
    "bot_farm": 30.0,
    "friendly_fraud": 40.0,
}


@dataclass(frozen=True)
class LabelConfig:
    maturity_days: int = 60  # after this, an approved event with no fraud report is legit
    delay_sigma: float = 0.6
    review_delay_days: float = 1.0
    seed: int = 101


def assign_labels(
    events: pd.DataFrame, decisions: np.ndarray, cfg: LabelConfig | None = None
) -> pd.DataFrame:
    """When, if ever, each event's label becomes known to the business.

    * reviewed: an analyst labels it about a day later (true label);
    * approved fraud: confirmed after a pattern-specific, log-normal delay capped at maturity;
    * approved legit: never reported; treated as legit once it is older than maturity;
    * declined: never labelled. The fraud was never attempted, or the customer left.
      This is the *selective labels* problem: the model's own declines hide its mistakes.
    """
    cfg = cfg or LabelConfig()
    rng = np.random.default_rng(cfg.seed)
    n = len(events)
    decisions = np.asarray(decisions)
    y = events["is_fraud"].to_numpy(int)
    medians = events["fraud_pattern"].map(LABEL_DELAY_MEDIAN_DAYS).fillna(30.0).to_numpy(float)
    delay = np.minimum(
        np.exp(np.log(medians) + cfg.delay_sigma * rng.standard_normal(n)), cfg.maturity_days
    )
    # A separate stream, so a shared prefix of events gets identical draws in every scenario.
    review_delay = np.random.default_rng(cfg.seed + 1).exponential(cfg.review_delay_days, n)
    days = np.full(n, np.nan)
    days[(decisions == "approve") & (y == 1)] = delay[(decisions == "approve") & (y == 1)]
    days[decisions == "review"] = review_delay[decisions == "review"]
    out = pd.DataFrame(index=events.index)
    out["decision"] = decisions
    out["label_delay_days"] = days
    out["label_at"] = events["timestamp"] + pd.to_timedelta(days, unit="D")
    return out
