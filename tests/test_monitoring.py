import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rde.monitoring.monitors import (
    DelayCurve,
    MonitorConfig,
    dispersion,
    poisson_cdf,
    poisson_sf,
    run_monitors,
)
from rde.monitoring.scenarios import (
    SCENARIOS,
    LabelConfig,
    ScenarioConfig,
    assign_labels,
    generate_scenario,
)
from rde.monitoring.study import (
    aggregate,
    load_study_config,
    run_study,
    with_seed,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study_cfg():
    return load_study_config(ROOT / "configs" / "monitoring_small.yaml")


@pytest.fixture(scope="module")
def scenario_events(study_cfg):
    return {
        s: generate_scenario(study_cfg.base.simulation, s, study_cfg.scenario) for s in SCENARIOS
    }


@pytest.fixture(scope="module")
def study(study_cfg):
    return run_study(study_cfg)


def test_history_is_identical_before_the_change(scenario_events):
    base = scenario_events["none"]
    for name, ev in scenario_events.items():
        pre_b, pre_e = base[~base["injected"]], ev[~ev["injected"]]
        cols = ["timestamp", "user_id", "amount", "is_fraud", "fraud_pattern"]
        pd.testing.assert_frame_equal(pre_b[cols], pre_e[cols], obj=name)


def test_incidents_only_add_what_they_claim(scenario_events):
    base = scenario_events["none"]
    post = {k: v[v["injected"]] for k, v in scenario_events.items()}
    assert len(post["ato_wave"]) > len(post["none"])
    ato = post["ato_wave"]["fraud_pattern"].eq("account_takeover").sum()
    assert ato > post["none"]["fraud_pattern"].eq("account_takeover").sum()
    assert post["evasive_bots"]["source_id"].eq("src_new_affiliate").any()
    partner = post["partner_campaign"][post["partner_campaign"]["user_id"].str.startswith("pu_")]
    assert len(partner) > 0 and partner["is_fraud"].sum() == 0
    assert not base["source_id"].isin(["src_new_affiliate", "src_partner"]).any()


def test_unknown_scenario_is_rejected(study_cfg):
    with pytest.raises(ValueError):
        generate_scenario(study_cfg.base.simulation, "meteor_strike")


def test_labels_follow_the_decision(scenario_events):
    ev = scenario_events["none"].head(4000)
    rng = np.random.default_rng(0)
    decisions = rng.choice(["approve", "review", "decline"], size=len(ev), p=[0.8, 0.1, 0.1])
    cfg = LabelConfig(maturity_days=60)
    lab = assign_labels(ev, decisions, cfg)
    fraud = ev["is_fraud"].to_numpy() == 1
    assert lab.loc[decisions == "decline", "label_at"].isna().all()
    assert lab.loc[(decisions == "approve") & ~fraud, "label_at"].isna().all()
    assert lab.loc[decisions == "review", "label_at"].notna().all()
    delays = lab.loc[(decisions == "approve") & fraud, "label_delay_days"]
    assert delays.notna().all() and (delays > 0).all() and (delays <= 60).all()
    assert (lab["label_at"].dropna() >= ev.loc[lab["label_at"].notna(), "timestamp"]).all()


def test_label_draws_are_stable_for_a_shared_prefix(scenario_events):
    ev = scenario_events["none"]
    d = np.full(len(ev), "approve")
    full = assign_labels(ev, d)["label_delay_days"]
    prefix = assign_labels(ev.head(3000), d[:3000])["label_delay_days"]
    pd.testing.assert_series_equal(full.head(3000), prefix)


def test_poisson_tails_match_brute_force():
    lam = 3.7
    pmf = [math.exp(-lam) * lam**k / math.factorial(k) for k in range(60)]
    for k in (0, 1, 3, 7, 12):
        assert poisson_cdf(k, lam) == pytest.approx(sum(pmf[: k + 1]), rel=1e-9)
        assert poisson_sf(k, lam) == pytest.approx(1 - sum(pmf[:k]), rel=1e-9, abs=1e-12)
    assert poisson_sf(0, lam) == 1.0


def test_delay_curve_is_a_cdf():
    curve = DelayCurve.fit([1, 2, 2, 5, 30])
    assert curve(0) == 0 and curve(30) == 1 and curve(2) == pytest.approx(0.6)
    assert np.all(np.diff(curve(np.linspace(0, 40, 50))) >= 0)
    with pytest.raises(ValueError):
        DelayCurve.fit([])


def test_dispersion_is_one_for_binomial_and_large_for_bursts():
    rng = np.random.default_rng(1)
    n = np.full(200, 1000)
    assert dispersion(rng.binomial(n, 0.03) / n, n) == pytest.approx(1.0, abs=0.35)
    bursty = np.where(rng.random(200) < 0.1, 0.2, 0.02)
    assert dispersion(rng.binomial(n, bursty) / n, n) > 10


def test_monitors_never_use_future_labels(study):
    run = study.runs["ato_wave"]
    frame = run.frame
    day = study.change_day + pd.Timedelta(days=10)
    T = day + pd.Timedelta(days=1)
    shuffled = frame.copy()
    future = shuffled["label_at"] > T
    shuffled.loc[future, "label_at"] = T + pd.Timedelta(days=1)  # move, but keep in the future
    features = pd.DataFrame(0.0, index=frame.index, columns=study.reference_obj.features.columns)
    a = run_monitors(frame, features, study.reference_obj, pd.DatetimeIndex([day]), 30)
    b = run_monitors(shuffled, features, study.reference_obj, pd.DatetimeIndex([day]), 30)
    cols = ["early_observed", "early_expected", "matured_observed", "matured_expected"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_study_end_to_end(study, tmp_path):
    names = [r["scenario"] for r in study.summary]
    assert names == list(SCENARIOS)
    none = study.summary[0]
    assert none["excess_cost_total"] == 0
    for row in study.summary:
        assert set(row["monitors"]) == {
            "feature_psi",
            "score_psi",
            "decision_mix",
            "early_labels",
            "matured_labels",
        }
    # Before the incident every scenario sees exactly the same monitor values.
    pre = study.runs["none"].daily.index < study.change_day
    for run in study.runs.values():
        pd.testing.assert_series_equal(
            run.daily.loc[pre, "early_expected"],
            study.runs["none"].daily.loc[pre, "early_expected"],
        )
    path = write_report(study, tmp_path, [study])
    text = path.read_text()
    assert "Monitoring under delayed labels" in text and "Robustness across 2" in text
    assert (tmp_path / "monitoring_summary.json").exists()
    rows = aggregate([study, study])
    assert all(r["seeds"] == 2 for r in rows)


def test_seed_override_changes_the_world(study_cfg):
    other = with_seed(study_cfg, 99)
    assert other.base.simulation.seed == 99 and other.labels.seed != study_cfg.labels.seed
    assert other.base.split == study_cfg.base.split


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "simulation: {n_days: 100}\nsplit: {train_end_day: 30, valid_end_day: 40,"
        " calib_end_day: 50}\nmonitoring: {scenario: {change_day: 40}}\n"
    )
    with pytest.raises(ValueError):
        load_study_config(bad)


def test_default_scenario_config_is_after_deployment():
    assert ScenarioConfig().change_day > 0
    assert MonitorConfig().persistence >= 1
