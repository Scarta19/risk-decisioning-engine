import numpy as np

from rde.modeling.calibration import fit_calibrator


def test_calibrator_fixes_overconfident_scores():
    rng = np.random.default_rng(1)
    true_p = rng.beta(0.6, 20, 30_000)
    y = rng.random(len(true_p)) < true_p
    raw = np.clip(true_p * 3, 0, 0.999)  # systematically overconfident
    cal, scores = fit_calibrator(raw, y)
    assert cal.method in {"platt", "isotonic"}
    assert scores[cal.method] < scores["identity"]
    p = cal.transform(raw)
    assert np.all((p > 0) & (p < 1))
    assert abs(p.mean() - y.mean()) < 0.01
