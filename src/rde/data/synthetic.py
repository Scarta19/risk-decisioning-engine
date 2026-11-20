"""Synthetic signup and purchase events with known fraud ground truth.

This is a parametric, agent-style simulator, not a copy of any real dataset. It encodes a
handful of fraud patterns that are widely described in public literature (card testing,
account takeover, bot/install farms, friendly fraud) and deliberately adds overlap with
legitimate behaviour so the problem is not trivially separable:

* legitimate users travel (geo mismatch), buy new phones (new device) and make bursts;
* some account takeovers come from the victim's own country;
* some bot farms randomise click-to-conversion time to look human;
* friendly fraud is indistinguishable from normal purchases by construction.

Because the generator knows the true label and the pattern of every event, evaluation can
report recall per pattern, which a real dataset rarely allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rde.config import SimulationConfig

COUNTRIES = np.array(["US", "CA", "MX", "CO", "BR", "GB"])
COUNTRY_WEIGHTS = np.array([0.45, 0.15, 0.10, 0.10, 0.10, 0.10])
PAYMENT_METHODS = np.array(["card", "wallet", "bank"])
PAYMENT_WEIGHTS = np.array([0.65, 0.25, 0.10])
# Relative activity by hour of day (UTC); legitimate traffic peaks in the afternoon/evening.
DIURNAL = np.array(
    [2, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 10, 10, 10, 9, 8, 6, 4, 3], dtype=float
)
DIURNAL = DIURNAL / DIURNAL.sum()

EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "event_type",
    "user_id",
    "device_id",
    "ip_prefix",
    "ip_country",
    "user_country",
    "source_id",
    "payment_method",
    "amount",
    "account_created_at",
    "click_to_conversion_sec",
    "is_fraud",
    "fraud_pattern",
]

DAY = 86_400.0


class _Sim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = pd.Timestamp(cfg.start, tz="UTC").timestamp()
        self.t_end = self.t0 + cfg.n_days * DAY
        self.rows: list[tuple] = []
        self.scale = cfg.n_days * cfg.n_users / 10_000  # user-days, in units of 10k
        self._ids = {"d": 0, "ip": 0, "f": 0}
        n_src = cfg.n_sources
        self.sources = np.array([f"src_{i:03d}" for i in range(n_src)])
        self.source_weights = self.rng.dirichlet(np.full(n_src, 2.0))
        self.bad_sources = self.rng.choice(self.sources, size=cfg.n_bad_sources, replace=False)

    # -- id helpers -------------------------------------------------------------------
    def _new(self, kind: str) -> str:
        self._ids[kind] += 1
        prefix = {"d": "dev", "ip": "ip", "f": "fu"}[kind]
        return f"{prefix}_{self._ids[kind]:07d}"

    def _hour_time(self, day_start: float) -> float:
        hour = self.rng.choice(24, p=DIURNAL)
        return day_start + hour * 3600 + self.rng.uniform(0, 3600)

    def _emit(
        self,
        ts,
        etype,
        user,
        device,
        ip,
        ip_cc,
        user_cc,
        src,
        pm,
        amount,
        created,
        ctc,
        fraud,
        pattern,
    ):
        if self.t0 <= ts < self.t_end:
            self.rows.append(
                (
                    ts,
                    etype,
                    user,
                    device,
                    ip,
                    ip_cc,
                    user_cc,
                    src,
                    pm,
                    float(amount),
                    created,
                    ctc,
                    int(fraud),
                    pattern,
                )
            )

    def _ctc_human(self) -> float:
        return float(self.rng.lognormal(np.log(900), 1.2))

    # -- legitimate population ------------------------------------------------------
    def legit(self) -> list[dict]:
        cfg, rng = self.cfg, self.rng
        users = []
        for i in range(cfg.n_users):
            is_new = rng.random() < cfg.new_user_share
            created = (
                rng.uniform(self.t0, self.t_end - DAY)
                if is_new
                else self.t0 - rng.uniform(1, 900) * DAY
            )
            u = {
                "id": f"u_{i:06d}",
                "country": rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS),
                "devices": [self._new("d") for _ in range(rng.integers(1, 3))],
                "ips": [self._new("ip") for _ in range(rng.integers(1, 4))],
                "pm": rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS),
                "mu": rng.normal(3.5, 0.6),
                "rate": rng.gamma(1.2, 0.06),
                "created": created,
            }
            users.append(u)
            if is_new:
                src = rng.choice(self.sources, p=self.source_weights)
                self._emit(
                    created,
                    "signup",
                    u["id"],
                    u["devices"][0],
                    u["ips"][0],
                    u["country"],
                    u["country"],
                    src,
                    None,
                    0.0,
                    created,
                    self._ctc_human(),
                    0,
                    "legit",
                )
            active_from = max(created + 600, self.t0)
            n = rng.poisson(u["rate"] * (self.t_end - active_from) / DAY)
            for _ in range(n):
                day = rng.uniform(active_from, self.t_end)
                day_start = self.t0 + np.floor((day - self.t0) / DAY) * DAY
                ts = max(self._hour_time(day_start), active_from)
                device = rng.choice(u["devices"])
                if rng.random() < 0.02:  # legitimately new phone / browser
                    device = self._new("d")
                    u["devices"].append(device)
                ip = rng.choice(u["ips"]) if rng.random() > 0.05 else self._new("ip")
                ip_cc = u["country"] if rng.random() > 0.03 else rng.choice(COUNTRIES)
                amount = round(float(rng.lognormal(u["mu"], 0.5)), 2)
                fraud = rng.random() < cfg.friendly_fraud_rate
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
                    int(fraud),
                    "friendly_fraud" if fraud else "legit",
                )
                # Occasional legitimate burst (e.g. buying several items in one session).
                if rng.random() < 0.03:
                    for k in range(rng.integers(1, 4)):
                        self._emit(
                            ts + 60 * (k + 1) * rng.uniform(1, 5),
                            "purchase",
                            u["id"],
                            device,
                            ip,
                            ip_cc,
                            u["country"],
                            "organic",
                            u["pm"],
                            round(float(rng.lognormal(u["mu"] - 0.5, 0.5)), 2),
                            u["created"],
                            None,
                            0,
                            "legit",
                        )
        return users

    # -- fraud patterns -------------------------------------------------------------
    def card_testing(self) -> None:
        cfg, rng = self.cfg, self.rng
        for _ in range(rng.poisson(cfg.card_testing_campaigns_per_10k_users_day * self.scale)):
            start = rng.uniform(self.t0, self.t_end)
            devices = [self._new("d") for _ in range(rng.integers(1, 3))]
            ip = self._new("ip")
            ip_cc = rng.choice(COUNTRIES)
            for _acct in range(rng.integers(1, 4)):
                user, cc = self._new("f"), rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
                created = start + rng.uniform(0, 600)
                src = rng.choice(self.sources, p=self.source_weights)
                self._emit(
                    created,
                    "signup",
                    user,
                    devices[0],
                    ip,
                    ip_cc,
                    cc,
                    src,
                    None,
                    0.0,
                    created,
                    self._ctc_human(),
                    1,
                    "card_testing",
                )
                ts = created + rng.uniform(60, 600)
                for _ in range(rng.integers(5, 20)):
                    ts += rng.exponential(90)
                    self._emit(
                        ts,
                        "purchase",
                        user,
                        rng.choice(devices),
                        ip,
                        ip_cc,
                        cc,
                        "organic",
                        "card",
                        round(float(rng.lognormal(np.log(4), 0.7)), 2),
                        created,
                        None,
                        1,
                        "card_testing",
                    )
                for _ in range(rng.integers(0, 3)):  # cash-out on a card that worked
                    ts += rng.exponential(600)
                    self._emit(
                        ts,
                        "purchase",
                        user,
                        rng.choice(devices),
                        ip,
                        ip_cc,
                        cc,
                        "organic",
                        "card",
                        round(float(rng.lognormal(np.log(150), 0.6)), 2),
                        created,
                        None,
                        1,
                        "card_testing",
                    )

    def account_takeover(self, users: list[dict]) -> None:
        cfg, rng = self.cfg, self.rng
        established = [u for u in users if u["created"] < self.t0]
        for _ in range(rng.poisson(cfg.ato_incidents_per_10k_users_day * self.scale)):
            u = established[rng.integers(len(established))]
            ts = rng.uniform(self.t0, self.t_end)  # attackers ignore the diurnal pattern
            if rng.random() < 0.25:  # session hijack: victim's own device and home network
                device, ip, ip_cc = rng.choice(u["devices"]), rng.choice(u["ips"]), u["country"]
            else:
                device, ip = self._new("d"), self._new("ip")
                ip_cc = rng.choice(COUNTRIES) if rng.random() < 0.6 else u["country"]
            for _ in range(rng.integers(1, 5)):
                ts += rng.exponential(1200)
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
                    round(float(rng.lognormal(np.log(250), 0.7)), 2),
                    u["created"],
                    None,
                    1,
                    "account_takeover",
                )

    def bot_farms(self) -> None:
        cfg, rng = self.cfg, self.rng
        for _ in range(rng.poisson(cfg.bot_campaigns_per_10k_users_day * self.scale)):
            src = rng.choice(self.bad_sources)
            start = rng.uniform(self.t0, self.t_end)
            duration = rng.uniform(1, 6) * 3600
            devices = [self._new("d") for _ in range(rng.integers(5, 21))]
            ips = [self._new("ip") for _ in range(rng.integers(2, 8))]
            cc = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
            evasive = rng.random() < 0.3  # mimics human click-to-conversion timing
            for _ in range(rng.integers(20, 100)):
                ts = start + rng.uniform(0, duration)
                ctc = self._ctc_human() if evasive else float(rng.lognormal(np.log(8), 0.8))
                self._emit(
                    ts,
                    "signup",
                    self._new("f"),
                    rng.choice(devices),
                    rng.choice(ips),
                    cc,
                    cc,
                    src,
                    None,
                    0.0,
                    ts,
                    ctc,
                    1,
                    "bot_farm",
                )

    def run(self) -> pd.DataFrame:
        users = self.legit()
        self.card_testing()
        self.account_takeover(users)
        self.bot_farms()
        df = pd.DataFrame(self.rows, columns=EVENT_COLUMNS[1:])
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        df.insert(0, "event_id", [f"evt_{i:08d}" for i in range(len(df))])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["account_created_at"] = pd.to_datetime(df["account_created_at"], unit="s", utc=True)
        return df[EVENT_COLUMNS]


def generate_events(cfg: SimulationConfig) -> pd.DataFrame:
    """Generate a time-ordered event table. Deterministic for a given config."""
    return _Sim(cfg).run()
