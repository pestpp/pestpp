"""ci subset of the structural noise stress test (structnoise_stress.py).

a linear model with a missing process, so what each correction should add is known exactly.
four settings, a few arms each, about 15 short pestpp-ies runs through the external run
manager (numpy forward model, no model binary).  the thresholds come from the full sweep, with
room for platform differences in the draws.
"""
import os
import shutil
import numpy as np

import structnoise_stress as sns


def _out_d():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "structnoise_stress_ci")


def _first(caps, arm, col):
    return float(caps.loc[(caps.arm == arm) & (caps.capture == 1), col].iloc[0])


def _last(caps, arm, col):
    c = caps.loc[caps.arm == arm]
    return float(c.loc[c.capture == c.capture.max(), col].iloc[0])


def _fin(fins, arm, col):
    return float(fins.loc[fins.arm == arm, col].iloc[0])


def structnoise_theory_test():
    """the known answers themselves: no structure means nothing to recover, p_eff is the number
    of coefficients when the prior is weak, and the one-capture em excess is short of the
    structure by about p_eff / n"""
    th0 = sns.theory(dict(sns.base_config(), amp_ratio=0.0))
    assert th0["s2_struct"].max() < 1.0e-20
    cfg = sns.base_config()
    th = sns.theory(cfg)
    assert abs(th["p_eff"] - cfg["k"]) < 0.05, th["p_eff"]
    ratio = th["excess_first"].mean() / th["s2_struct"].mean()
    assert abs(ratio - (1.0 - th["p_eff"] / cfg["n_obs"])) < 0.05, ratio


def structnoise_base_test():
    """the base setting: structure 3x the noise.
    - every correction adds what the linear theory says at the first capture
    - em recovers (1 - p_eff/n) of the structure in one capture and climbs toward all of it
      with recapture
    - the default correction adds more than em: it counts the measurement noise again
    - the fraction scales the em correction linearly
    - any correction cuts the final actual phi well below no correction"""
    caps, fins = sns.run_setting("base", ["none", "def_once", "em_once", "em_recap", "em_recap_x2"],
                                 _out_d())
    for arm in ["def_once", "em_once", "em_recap", "em_recap_x2"]:
        r = _first(caps, arm, "first_capture_over_theory")
        assert 0.97 < r < 1.03, (arm, r)
    p_eff, n = float(caps.p_eff.iloc[0]), float(caps.n_obs.iloc[0])
    em1 = _first(caps, "em_once", "added_over_struct")
    assert abs(em1 - (1.0 - p_eff / n)) < 0.05, em1
    em_last = _last(caps, "em_recap", "added_over_struct")
    assert em_last > em1, (em1, em_last)
    assert 0.95 < em_last < 1.15, em_last
    def1 = _first(caps, "def_once", "added_over_struct")
    assert def1 > em1 + 0.1, (def1, em1)
    x2 = _first(caps, "em_recap_x2", "added_mean") / _first(caps, "em_recap", "added_mean")
    assert abs(x2 - 2.0) < 0.04, x2
    phi_none = _fin(fins, "none", "final_phi_actual_mean")
    for arm in ["def_once", "em_once", "em_recap"]:
        assert _fin(fins, arm, "final_phi_actual_mean") < phi_none / 3.0, arm


def structnoise_nostruct_test():
    """no missing process: there is nothing structural to find.  the default correction still
    adds about the noise variance (the measurement error in the observed values, counted again);
    em adds a fraction of that, only where sampling puts the residuals above the noise"""
    caps, fins = sns.run_setting("amp0", ["def_once", "em_once"], _out_d())
    d = _first(caps, "def_once", "added_over_sig2")
    e = _first(caps, "em_once", "added_over_sig2")
    assert 0.6 < d < 1.2, d
    assert e < 0.4, e
    assert e < 0.5 * d, (e, d)
    assert _first(caps, "em_once", "frac_obs_corrected") < 0.5


def structnoise_weights_test():
    """weights 3x looser than the noise: em measures the excess against the noise realizations,
    not the weights.  if it used the weights (a variance 9x the noise) it would find almost no
    excess; it finds the same as with matching weights"""
    caps, fins = sns.run_setting("wts3x", ["em_once"], _out_d())
    r = _first(caps, "em_once", "first_capture_over_theory")
    assert 0.97 < r < 1.03, r
    a = _first(caps, "em_once", "added_over_struct")
    assert 0.85 < a < 1.05, a


def structnoise_phifactor_test():
    """phi factors rebalance the two halves of the record 0.8 / 0.2 at the start, so the weights
    no longer match the noise (they go from 10 to about 13 and 6).  em still measures against
    the noise, and the de-weighting starts from the rebalanced weights:
    - every correction adds what the linear theory says, with H from the rebalanced weights
    - em recovers most of the structure and climbs with recapture, as with matching weights
    - the default still adds more than em
    - any correction cuts the final actual phi well below no correction"""
    caps, fins = sns.run_setting("phifac", ["none", "def_once", "em_once", "em_recap"], _out_d())
    wmin, wmax = _fin(fins, "em_once", "base_weight_min"), _fin(fins, "em_once", "base_weight_max")
    assert wmax / wmin > 1.5, (wmin, wmax)
    for arm in ["def_once", "em_once", "em_recap"]:
        r = _first(caps, arm, "first_capture_over_theory")
        assert 0.97 < r < 1.03, (arm, r)
    em1 = _first(caps, "em_once", "added_over_struct")
    assert 0.85 < em1 < 1.05, em1
    assert _last(caps, "em_recap", "added_over_struct") > em1
    assert _first(caps, "def_once", "added_over_struct") > em1 + 0.1
    phi_none = _fin(fins, "none", "final_phi_actual_mean")
    for arm in ["def_once", "em_once", "em_recap"]:
        assert _fin(fins, arm, "final_phi_actual_mean") < phi_none / 3.0, arm


if __name__ == "__main__":
    if os.path.exists(_out_d()):
        shutil.rmtree(_out_d())
    structnoise_theory_test()
    structnoise_base_test()
    structnoise_nostruct_test()
    structnoise_weights_test()
    structnoise_phifactor_test()
