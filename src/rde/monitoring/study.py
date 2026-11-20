"""Monitoring study: how fast does each monitor notice each kind of drift, and at what cost?

The model is trained once on the shared history (identical in every scenario), deployed at
the end of the calibration window and then left alone. Each scenario replays the production
period with its own incident, assigns labels with realistic delays and runs every monitor
day by day. Because all scenarios share the same history and the same pre-change events,
``scenario cost - none cost`` is the money the incident actually cost.

Simplification, stated plainly: training, calibration and the reference statistics use
labels of windows that would not be fully matured at deployment in real life. In practice
you would train on data older than the maturity window; that changes timing, not the method.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rde.config import Config, config_from_dict
from rde.decision.policy import realized_cost
from rde.features.state import build_feature_matrix
from rde.monitoring.monitors import (
    MONITORS,
    DelayCurve,
    MonitorConfig,
    Reference,
    dispersion,
    run_monitors,
)
from rde.monitoring.scenarios import (
    SCENARIOS,
    LabelConfig,
    ScenarioConfig,
    assign_labels,
    generate_scenario,
)
from rde.pipeline import run_pipeline

log = logging.getLogger(__name__)
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class StudyConfig:
    base: Config
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    monitors: MonitorConfig = field(default_factory=MonitorConfig)


def load_study_config(path: str | Path) -> StudyConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    mon = raw.pop("monitoring", {}) or {}
    base = config_from_dict(raw)
    sc_raw = dict(mon.get("scenario") or {})
    if "partner_signups_per_blast" in sc_raw:
        sc_raw["partner_signups_per_blast"] = tuple(sc_raw["partner_signups_per_blast"])
    cfg = StudyConfig(
        base=base,
        scenario=ScenarioConfig(**sc_raw),
        labels=LabelConfig(**(mon.get("labels") or {})),
        monitors=MonitorConfig(**(mon.get("monitors") or {})),
    )
    s, n = base.split, base.simulation.n_days
    if not s.calib_end_day < cfg.scenario.change_day < n:
        raise ValueError("change_day must fall after deployment (calib_end_day) and before n_days")
    if s.calib_end_day <= cfg.labels.maturity_days:
        raise ValueError("deployment must come after maturity_days of history (delay curve)")
    return cfg


@dataclass
class ScenarioRun:
    name: str
    daily: pd.DataFrame  # monitor statistics and alarms by day
    cost: pd.Series  # realized cost by day (true labels, known only to the simulator)
    frame: pd.DataFrame  # production events with scores, decisions and label times


@dataclass
class StudyResult:
    config: StudyConfig
    deploy_day: pd.Timestamp
    change_day: pd.Timestamp
    reference: dict[str, Any]
    runs: dict[str, ScenarioRun]
    summary: list[dict[str, Any]]
    reference_obj: Reference | None = None  # the monitors' reference, for inspection


def _day0(events: pd.DataFrame) -> pd.Timestamp:
    return events["timestamp"].min().normalize()


def run_study(cfg: StudyConfig, scenarios=SCENARIOS) -> StudyResult:
    sim = cfg.base.simulation
    log.info("Simulating the shared history and the no-change scenario")
    base_events = generate_scenario(sim, "none", cfg.scenario)
    result = run_pipeline(cfg.base, base_events.drop(columns="injected"))
    engine, policy = result.engine, result.engine.policy

    day0 = _day0(base_events)
    deploy = day0 + cfg.base.split.calib_end_day * DAY
    change = day0 + cfg.scenario.change_day * DAY
    ts = base_events["timestamp"]

    # Reference window = calibration window, the most recent data before deployment.
    ref_mask = ((ts >= day0 + cfg.base.split.valid_end_day * DAY) & (ts < deploy)).to_numpy()
    ev_ref = base_events[ref_mask]
    p_ref = result.scores[ref_mask].to_numpy(float)
    d_ref = policy.decide(p_ref, ev_ref["amount"].to_numpy(float), ev_ref["event_type"].to_numpy())
    appr = d_ref == "approve"
    y_ref = ev_ref["is_fraud"].to_numpy(int)
    approved_calibration = float(y_ref[appr].sum() / max(p_ref[appr].sum(), 1e-9))

    # Delay curve from history that is fully matured at deployment. Before the model existed
    # every event was approved, so every fraud eventually produced a label.
    hist = base_events[(ts < deploy - cfg.labels.maturity_days * DAY).to_numpy()]
    hist_labels = assign_labels(hist, np.full(len(hist), "approve"), cfg.labels)
    curve = DelayCurve.fit(hist_labels.loc[hist["is_fraud"] == 1, "label_delay_days"])

    # Decision rates and their day-to-day dispersion, from the 30 days before deployment
    # (early-stopping and calibration windows; the training window is in-sample).
    rate_mask = ((ts >= day0 + cfg.base.split.train_end_day * DAY) & (ts < deploy)).to_numpy()
    ev_rate = base_events[rate_mask]
    p_rate = result.scores[rate_mask].to_numpy(float)
    d_rate = pd.Series(
        policy.decide(p_rate, ev_rate["amount"].to_numpy(float), ev_rate["event_type"].to_numpy()),
        index=ev_rate.index,
    )
    by_day = d_rate.groupby(ev_rate["timestamp"].dt.floor("D"))
    n_day = by_day.size()
    dec_day = by_day.apply(lambda s: (s == "decline").mean())
    rev_day = by_day.apply(lambda s: (s == "review").mean())

    reference = Reference(
        features=result.features[ref_mask],
        scores=p_ref,
        decline_rate=float((d_rate == "decline").mean()),
        review_rate=float((d_rate == "review").mean()),
        approved_calibration=approved_calibration,
        delay_curve=curve,
        decline_dispersion=dispersion(dec_day, n_day),
        review_dispersion=dispersion(rev_day, n_day),
    )
    days = pd.date_range(deploy, day0 + (sim.n_days - 1) * DAY, freq="D")

    runs: dict[str, ScenarioRun] = {}
    for name in scenarios:
        log.info("Scenario %s: features, decisions, labels, monitors", name)
        if name == "none":
            events, X, p = base_events, result.features, result.scores.to_numpy(float)
        else:
            events = generate_scenario(sim, name, cfg.scenario)
            X = build_feature_matrix(events)
            p = engine.score(X)
        prod = (events["timestamp"] >= deploy).to_numpy()
        frame = events[prod].copy()
        frame["p"] = p[prod]
        frame["decision"] = policy.decide(
            frame["p"].to_numpy(), frame["amount"].to_numpy(float), frame["event_type"].to_numpy()
        )
        labels = assign_labels(frame, frame["decision"].to_numpy(), cfg.labels)
        frame["label_at"] = labels["label_at"]
        daily = run_monitors(
            frame, X[prod], reference, days, cfg.labels.maturity_days, cfg.monitors
        )
        cost = pd.Series(
            realized_cost(
                policy,
                frame["decision"].to_numpy(),
                frame["is_fraud"].to_numpy(int),
                frame["amount"].to_numpy(float),
                frame["event_type"].to_numpy(),
            ),
            index=frame.index,
        )
        daily_cost = (
            cost.groupby(frame["timestamp"].dt.floor("D")).sum().reindex(days, fill_value=0)
        )
        runs[name] = ScenarioRun(name, daily, daily_cost, frame)

    summary = _summarise(runs, deploy, change)
    ref_info = {
        "window": _span(ev_ref),
        "events": int(ref_mask.sum()),
        "rate_window": _span(ev_rate),
        "decline_rate": reference.decline_rate,
        "review_rate": reference.review_rate,
        "decline_dispersion": reference.decline_dispersion,
        "review_dispersion": reference.review_dispersion,
        "approved_calibration": approved_calibration,
        "delay_curve_n": len(curve.delays),
        "delay_quantiles_days": {
            q: float(np.quantile(curve.delays, q)) for q in (0.25, 0.5, 0.75, 0.9)
        },
    }
    return StudyResult(cfg, deploy, change, ref_info, runs, summary, reference)


def _span(events: pd.DataFrame) -> str:
    return f"{events['timestamp'].min().date()} to {events['timestamp'].max().date()}"


def _summarise(runs: dict[str, ScenarioRun], deploy, change) -> list[dict[str, Any]]:
    """Detection lag and money lost before detection, per scenario and monitor.

    An alarm after the change only counts as a detection if the ``none`` scenario does not
    raise the same alarm on the same day; otherwise it is noise the incident did not cause.
    """
    base = runs.get("none")
    base_cost = base.cost if base is not None else None
    out = []
    for name, run in runs.items():
        d = run.daily
        pre = d.index < change
        post = ~pre
        excess = (run.cost - base_cost) if base_cost is not None else run.cost * 0
        row: dict[str, Any] = {
            "scenario": name,
            "post_change_events": int((run.frame["timestamp"] >= change).sum()),
            "excess_cost_total": float(excess[post].sum()),
            "excess_cost_per_day": float(excess[post].mean()),
            "monitors": {},
        }
        for m in MONITORS:
            alarm = d[f"alarm_{m}"]
            caused = alarm.to_numpy() & post
            if base is not None and name != "none":
                caused &= ~base.daily[f"alarm_{m}"].reindex(d.index, fill_value=False).to_numpy()
            first = alarm[caused].index.min()
            detected = pd.notna(first)
            lag = int((first - change) / DAY) if detected else None
            sig_col = {"early_labels": "early_signal", "matured_labels": "matured_signal"}.get(m)
            signal = str(d.loc[first, sig_col]) if detected and sig_col else None
            row["monitors"][m] = {
                "false_alarm_days_before_change": int(alarm[pre].sum()),
                "alarm_days_after_change": int(alarm[post].sum()),
                "incident_alarm_days": int(caused.sum()),
                "first_alarm_lag_days": lag,
                "first_alarm_signal": signal,
                "excess_cost_before_alarm": float(
                    excess[(excess.index >= change) & (excess.index <= first)].sum()
                    if detected
                    else excess[post].sum()
                ),
            }
        out.append(row)
    return out


# -- report -----------------------------------------------------------------------------

LABELS = {
    "feature_psi": "Feature PSI",
    "score_psi": "Score PSI",
    "decision_mix": "Decision mix",
    "early_labels": "Early labels (age-adjusted)",
    "matured_labels": "Matured labels",
}


def _fmt_lag(m: dict[str, Any], scenario: str) -> str:
    if scenario == "none":
        n = m["false_alarm_days_before_change"] + m["alarm_days_after_change"]
        return "no alarm" if n == 0 else f"{n} false alarm days"
    lag = m["first_alarm_lag_days"]
    txt = "not detected" if lag is None else f"day {lag}"
    if m.get("first_alarm_signal"):
        txt += f" ({m['first_alarm_signal']})"
    if m["false_alarm_days_before_change"]:
        txt += f" ({m['false_alarm_days_before_change']} false before)"
    return txt


def _figure(result: StudyResult, out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = [n for n in result.runs if n != "none"]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.6 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        d = result.runs[name].daily
        ax.plot(d.index, d["early_ratio"].clip(upper=6), label="early labels: observed / expected")
        ax.plot(
            d.index, d["matured_ratio"].clip(upper=6), label="matured labels: observed / expected"
        )
        mix = np.maximum(d["decline_z"].abs(), d["review_z"].abs()).clip(upper=12) / 2
        ax.plot(d.index, mix, label="decision mix: max |z| / 2", alpha=0.7)
        ax.axvline(result.change_day, color="k", ls="--", lw=1)
        ax.axhline(1.0, color="grey", lw=0.5)
        ax.set_title(name)
        ax.set_ylim(-1, 6.5)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = out_dir / "figures" / "monitor_signals.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "figures/monitor_signals.png"


def with_seed(cfg: StudyConfig, seed: int) -> StudyConfig:
    """The same study on a different simulated world (and label-delay draws)."""
    base = replace(cfg.base, simulation=replace(cfg.base.simulation, seed=seed))
    return replace(cfg, base=base, labels=replace(cfg.labels, seed=seed + 1000))


def aggregate(results: list[StudyResult]) -> list[dict[str, Any]]:
    """Detection rate and median lag per scenario and monitor across seeds."""
    rows = []
    for scenario in results[0].runs:
        for m in MONITORS:
            stats = [
                next(r for r in res.summary if r["scenario"] == scenario)["monitors"][m]
                for res in results
            ]
            lags = [
                st["first_alarm_lag_days"] for st in stats if st["first_alarm_lag_days"] is not None
            ]
            false_days = sum(st["false_alarm_days_before_change"] for st in stats)
            if scenario == "none":
                false_days += sum(st["alarm_days_after_change"] for st in stats)
            rows.append(
                {
                    "scenario": scenario,
                    "monitor": m,
                    "seeds": len(stats),
                    "detected": len(lags) if scenario != "none" else None,
                    "median_lag_days": float(np.median(lags)) if lags else None,
                    "false_alarm_days": int(false_days),
                }
            )
    return rows


def _agg_cell(row: dict[str, Any]) -> str:
    if row["scenario"] == "none":
        return (
            "no alarm" if not row["false_alarm_days"] else f"{row['false_alarm_days']} false days"
        )
    if not row["detected"]:
        return f"0/{row['seeds']}"
    return f"{row['detected']}/{row['seeds']}, median day {row['median_lag_days']:.0f}"


def write_report(
    result: StudyResult, out_dir: str | Path, others: list[StudyResult] | None = None
) -> Path:
    """Write the report for ``result``; ``others`` (more seeds) add a robustness table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    summary = {
        "deploy_day": str(result.deploy_day.date()),
        "change_day": str(result.change_day.date()),
        "maturity_days": cfg.labels.maturity_days,
        "reference": result.reference,
        "scenarios": result.summary,
        "config": {
            "scenario": asdict(cfg.scenario),
            "labels": asdict(cfg.labels),
            "monitors": asdict(cfg.monitors),
            "simulation": asdict(cfg.base.simulation),
            "split": asdict(cfg.base.split),
        },
    }
    agg = aggregate([result, *others]) if others else None
    if agg:
        seeds = [result.config.base.simulation.seed] + [
            o.config.base.simulation.seed for o in others
        ]
        summary["robustness"] = {"seeds": seeds, "by_scenario_monitor": agg}
    (out_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for name, run in result.runs.items():
        run.daily.to_csv(out_dir / f"daily_{name}.csv")
    fig = _figure(result, out_dir)

    lines = [
        "# Monitoring under delayed labels",
        "",
        f"Deployed {summary['deploy_day']}; incident starts {summary['change_day']}; fraud labels"
        f" mature after {cfg.labels.maturity_days} days. Synthetic data, illustrative costs.",
        "",
        "## When each monitor first raises an alarm (days after the incident starts)",
        "",
        "| Scenario | Excess cost / day | " + " | ".join(LABELS[m] for m in MONITORS) + " |",
        "|---|---:|" + "---|" * len(MONITORS),
    ]
    for row in result.summary:
        cells = [_fmt_lag(row["monitors"][m], row["scenario"]) for m in MONITORS]
        lines.append(
            f"| {row['scenario']} | ${row['excess_cost_per_day']:,.0f} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Money lost before the first alarm",
        "",
        "| Scenario | " + " | ".join(LABELS[m] for m in MONITORS) + " |",
        "|---|" + "---:|" * len(MONITORS),
    ]
    for row in result.summary:
        if row["scenario"] == "none":
            continue
        cells = [f"${row['monitors'][m]['excess_cost_before_alarm']:,.0f}" for m in MONITORS]
        lines.append(f"| {row['scenario']} | " + " | ".join(cells) + " |")
    if agg:
        lines += [
            "",
            f"## Robustness across {agg[0]['seeds']} simulated worlds",
            "",
            "Detected in how many seeds, and the median first-alarm day when detected.",
            "",
            "| Scenario | " + " | ".join(LABELS[m] for m in MONITORS) + " |",
            "|---|" + "---|" * len(MONITORS),
        ]
        for scenario in result.runs:
            cells = [_agg_cell(r) for r in agg if r["scenario"] == scenario]
            lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
    ref = result.reference
    lines += [
        "",
        "## Reference at deployment",
        "",
        f"* Feature and score reference: {ref['window']}, {ref['events']:,} events.",
        f"* Decision-rate reference: {ref['rate_window']}; decline rate {ref['decline_rate']:.2%}"
        f" (dispersion {ref['decline_dispersion']:.1f}x binomial), review rate"
        f" {ref['review_rate']:.2%} (dispersion {ref['review_dispersion']:.1f}x).",
        f"* Approved-event calibration factor {ref['approved_calibration']:.2f}"
        " (observed / predicted fraud among approved events).",
        f"* Label-delay curve from {ref['delay_curve_n']} matured fraud labels; median"
        f" {ref['delay_quantiles_days'][0.5]:.0f} days, 90th percentile"
        f" {ref['delay_quantiles_days'][0.9]:.0f} days.",
    ]
    if fig:
        lines += ["", f"![Monitor signals by scenario]({fig})"]
    path = out_dir / "monitoring.md"
    path.write_text("\n".join(lines) + "\n")
    return path
