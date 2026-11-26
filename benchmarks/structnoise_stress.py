"""stress test for the structural noise correction in pestpp-ies, on a problem with known structure.

stage 1: a linear model with a missing process.  the model is a sum of k legendre polynomials in
time with a gaussian prior on the coefficients.  the truth is the same model at known
coefficients plus a process the model cant represent (a sine, or an uncorrelated random signal),
and the observed values are the truth plus measurement noise of known size.  because the model is
linear, the structural residual per observation, p_eff and what each method should add at the
first capture are all known exactly, so every arm is scored against numbers, not by eye.

the model runs through pestpp's external run manager (pestpp-ies case.pst /e): pestpp writes each
batch of runs to case.rns and calls `python forward_run.py` once per batch.

    python structnoise_stress.py            # every setting and arm, in parallel
    python structnoise_stress.py --quick    # the base setting only
    python structnoise_stress.py --report   # rebuild the csv and pdf from finished runs

results: structnoise_stress/summary.csv (one row per setting x arm x capture),
structnoise_stress/final.csv (one row per setting x arm) and structnoise_stress/structnoise_stress.pdf.
"""
import os
import sys
import glob
import json
import shutil
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- the model

def base_config():
    """the base setting.  every other setting changes one entry of this"""
    return {
        "n_obs": 60,            # observation times on [0, 1]
        "k": 4,                 # legendre terms in the model
        "prior_sd": 1.0,        # prior sd of every coefficient
        "sigma": 0.1,           # measurement noise sd
        "amp_ratio": 3.0,       # amplitude of the missing process / sigma
        "shape": "sine",        # "sine" (correlated) or "random" (uncorrelated)
        "period": 0.23,         # sine period, as a fraction of the record
        "weight_sd_factor": 1.0,  # weight = 1 / (factor * sigma); noise is always drawn with sigma
        "n_reals": 50,
        "first_cycle": 4,       # iterations before the first reinflation (and capture)
        "cycle": 4,             # iterations in each later cycle
        "n_cycles": 5,          # cycles in all, so n_cycles - 1 captures
        "truth_seed": 7,
        "noise_seed": 11,
        "ens_seed": 13,
        # {tag: factor} for ies_phi_factor_file: the record is split into two groups (first and
        # second half, "reca" and "recb") and pestpp-ies rebalances their weights at the start,
        # so the weights no longer match the noise.  None leaves one group and the user weights
        "phi_factors": None,
        # with phi_factors: True gives every realization its own split of the two halves (reca
        # drawn uniformly on 0.2-0.8, recb = 1 - reca) through ies_phi_factors_by_real, so the
        # weights differ by realization as well as from the noise
        "phi_factors_by_real": False,
    }


def times(cfg):
    return np.linspace(0.0, 1.0, cfg["n_obs"])


def basis(cfg, t):
    return np.polynomial.legendre.legvander(2.0 * np.asarray(t) - 1.0, cfg["k"] - 1)


def par_names(cfg):
    return ["c{0:02d}".format(j) for j in range(cfg["k"])]


def obs_names(cfg):
    return ["y_{0:03d}".format(i) for i in range(cfg["n_obs"])] + ["fc_shared", "fc_struct"]


def forecast_rows(cfg):
    """fc_shared is a fixed combination of the coefficients (the missing process cant touch it);
    fc_struct is the value between two observation times, where the truth also has the missing
    process"""
    c = 1.0 / (1.0 + np.arange(cfg["k"]))
    t_mid = 0.5 * (times(cfg)[cfg["n_obs"] // 2] + times(cfg)[cfg["n_obs"] // 2 + 1])
    return c, basis(cfg, [t_mid])[0], t_mid


def model_outputs(cfg, theta):
    y = basis(cfg, times(cfg)) @ theta
    c, b_mid, _ = forecast_rows(cfg)
    return np.concatenate([y, [c @ theta, b_mid @ theta]])


def missing_process(cfg, t):
    amp = cfg["amp_ratio"] * cfg["sigma"]
    if cfg["shape"] == "sine":
        return amp * np.sin(2.0 * np.pi * np.asarray(t) / cfg["period"] + 0.7)
    rng = np.random.default_rng(cfg["truth_seed"] + 1000)
    return amp * rng.standard_normal(len(np.atleast_1d(t)))


def truth(cfg):
    """true coefficients, true outputs (with the missing process) and the observed values"""
    theta = cfg["prior_sd"] * np.random.default_rng(cfg["truth_seed"]).standard_normal(cfg["k"])
    t = times(cfg)
    g = missing_process(cfg, t)
    y_true = basis(cfg, t) @ theta + g
    c, b_mid, t_mid = forecast_rows(cfg)
    if cfg["shape"] == "sine":
        g_mid = missing_process(cfg, [t_mid])[0]
    else:
        g_mid = 0.5 * (g[cfg["n_obs"] // 2] + g[cfg["n_obs"] // 2 + 1])
    fc = np.array([c @ theta, b_mid @ theta + g_mid])
    eps = cfg["sigma"] * np.random.default_rng(cfg["noise_seed"]).standard_normal(cfg["n_obs"])
    return theta, np.concatenate([y_true, fc]), y_true + eps, g


def run_forward_rns(ws=".", case="case"):
    """the model command: fill in every run of the batch in case.rns"""
    from pyemu.utils.helpers import RunStor
    with open(os.path.join(ws, "config.json")) as f:
        cfg = json.load(f)
    fname = os.path.join(ws, case + ".rns")
    _, pnames, onames = RunStor.file_info(fname)
    rs = RunStor(fname)
    df = rs.get_data()
    low = {n.lower(): n for n in pnames}
    pcols = [low[n] for n in par_names(cfg)]
    low = {n.lower(): n for n in onames}
    ocols = [low[n] for n in obs_names(cfg)]
    pv = df.loc[:, pcols].values.astype(float)
    df.loc[:, ocols] = np.array([model_outputs(cfg, p) for p in pv])
    df.loc[:, "run_status"] = 1
    rs.update(df)


# ---------------------------------------------------------------- what the answers should be

def theory(cfg, weights=None):
    """the known answers for the record observations.

    s2_struct: the part of the missing process no choice of coefficients can reach, squared
               (residual of projecting it onto the model's columns)
    p_eff:     trace of the hat matrix with the prior and the weights the solve uses
    s2_first:  expected uncentred mean square of the actual residuals at a converged first
               capture, for a linear ensemble smoother from the prior:
               r_k = (I - H)(d_obs - J theta_k) - H e_k
    sig2:      the noise variance (what em subtracts)
    weights:   the weights the solve uses, if not 1 / (weight_sd_factor * sigma) - with phi
               factors these are the rebalanced weights pestpp-ies saved
    """
    t = times(cfg)
    J = basis(cfg, t)
    theta_t, _, d_obs, g = truth(cfg)
    n = cfg["n_obs"]
    P = J @ np.linalg.pinv(J)
    s2_struct = ((np.eye(n) - P) @ g) ** 2
    sig2 = np.full(n, cfg["sigma"] ** 2)
    if weights is None:
        weights = np.full(n, 1.0 / (cfg["weight_sd_factor"] * cfg["sigma"]))
    ce = np.diag(1.0 / np.asarray(weights) ** 2)
    cp = np.eye(cfg["k"]) * cfg["prior_sd"] ** 2
    jcj = J @ cp @ J.T
    H = jcj @ np.linalg.inv(jcj + ce)
    IH = np.eye(n) - H
    mean_r = IH @ d_obs
    cov_r = IH @ jcj @ IH.T + H @ np.diag(sig2) @ H.T
    s2_first = mean_r ** 2 + np.diag(cov_r)
    return {"s2_struct": s2_struct, "sig2": sig2, "p_eff": float(np.trace(H)),
            "s2_first": s2_first, "excess_first": np.maximum(0.0, s2_first - sig2)}


# ---------------------------------------------------------------- settings and arms

def settings():
    """the base setting plus one knob changed at a time"""
    out = {"base": {}}
    for a in (0.0, 0.3, 1.0, 10.0):
        out["amp{0:g}".format(a)] = {"amp_ratio": a}
    for k in (2, 8, 20):
        out["k{0}".format(k)] = {"k": k}
    out["ne250"] = {"n_reals": 250}
    out["early"] = {"first_cycle": 1}
    out["wts3x"] = {"weight_sd_factor": 3.0}
    out["random"] = {"shape": "random"}
    out["phifac"] = {"phi_factors": {"reca": 0.8, "recb": 0.2}}
    out["phifac_real"] = {"phi_factors": {"reca": 0.8, "recb": 0.2}, "phi_factors_by_real": True}
    return out


def arms():
    """the options of each arm, on top of the shared schedule"""
    corr = {"ies_structnoise_deweight": True}
    return {
        "none":       {"ies_reinflate_structnoise_frac": 0.0},
        "def_once":   dict(corr, ies_reinflate_structnoise_frac=1.0),
        "def_recap":  dict(corr, ies_reinflate_structnoise_frac=-1.0),
        "em_once":    dict(corr, ies_reinflate_structnoise_frac=1.0, ies_structnoise_em=True),
        "em_recap":   dict(corr, ies_reinflate_structnoise_frac=-1.0, ies_structnoise_em=True),
        "em_recap_x0.5": dict(corr, ies_reinflate_structnoise_frac=-0.5, ies_structnoise_em=True),
        "em_recap_x2":   dict(corr, ies_reinflate_structnoise_frac=-2.0, ies_structnoise_em=True),
        # alternate: one iteration, then reinflate and recapture, every iteration
        "em_recap_every": dict(corr, ies_reinflate_structnoise_frac=-1.0, ies_structnoise_em=True,
                               ies_n_iter_reinflate=1),
    }


def arm_colors():
    return {"none": "0.5", "def_once": "#e69f00", "def_recap": "#d55e00", "em_once": "#56b4e9",
            "em_recap": "#0072b2", "em_recap_x0.5": "#009e73", "em_recap_x2": "#cc79a7",
            "em_recap_every": "#000000"}


def arm_frac(arm_opts):
    return abs(arm_opts["ies_reinflate_structnoise_frac"])


# ---------------------------------------------------------------- building and running

def build_template(cfg, t_d):
    """the control file, the shared prior and noise ensembles, and the model files"""
    import pyemu
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    os.makedirs(t_d)
    shutil.copy2(os.path.abspath(__file__), os.path.join(t_d, "structnoise_stress.py"))
    with open(os.path.join(t_d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(t_d, "forward_run.py"), "w") as f:
        f.write("import structnoise_stress\nif __name__ == '__main__':\n"
                "    structnoise_stress.run_forward_rns(case='case')\n")
    pn, on = par_names(cfg), obs_names(cfg)
    theta_t, outs_t, d_obs, _ = truth(cfg)
    with open(os.path.join(t_d, "pars.dat.tpl"), "w") as f:
        f.write("ptf ~\n")
        for p in pn:
            f.write("{0} ~ {0:^20} ~\n".format(p))
    with open(os.path.join(t_d, "outputs.dat.ins"), "w") as f:
        f.write("pif ~\n")
        for o in on:
            f.write("l1 w !{0}!\n".format(o))
    with open(os.path.join(t_d, "pars.dat"), "w") as f:
        for p in pn:
            f.write("{0} 0.0\n".format(p))
    with open(os.path.join(t_d, "outputs.dat"), "w") as f:
        for o, v in zip(on, model_outputs(cfg, np.zeros(cfg["k"]))):
            f.write("{0} {1}\n".format(o, v))
    pst = pyemu.Pst.from_io_files([os.path.join(t_d, "pars.dat.tpl")], [os.path.join(t_d, "pars.dat")],
                                  [os.path.join(t_d, "outputs.dat.ins")], [os.path.join(t_d, "outputs.dat")],
                                  pst_path=".")
    par = pst.parameter_data
    par.loc[pn, "partrans"] = "none"
    par.loc[pn, "parval1"] = 0.0
    par.loc[pn, "parlbnd"] = -10.0 * cfg["prior_sd"]
    par.loc[pn, "parubnd"] = 10.0 * cfg["prior_sd"]
    par.loc[pn, "pargp"] = "coef"
    obs = pst.observation_data
    rec = on[:cfg["n_obs"]]
    obs.loc[rec, "obsval"] = d_obs
    obs.loc[rec, "weight"] = 1.0 / (cfg["weight_sd_factor"] * cfg["sigma"])
    obs.loc[rec, "obgnme"] = "rec"
    if cfg.get("phi_factors"):
        obs.loc[rec[:cfg["n_obs"] // 2], "obgnme"] = "reca"
        obs.loc[rec[cfg["n_obs"] // 2:], "obgnme"] = "recb"
        if cfg.get("phi_factors_by_real"):
            frng = np.random.default_rng(cfg["ens_seed"] + 1)
            fa = frng.uniform(0.2, 0.8, cfg["n_reals"])
            pfr = pd.DataFrame({"reca": fa, "recb": 1.0 - fa},
                               index=[str(i) for i in range(cfg["n_reals"])])
            pfr.to_csv(os.path.join(t_d, "phi_factors.csv"))
        else:
            with open(os.path.join(t_d, "phi_factors.dat"), "w") as f:
                for tag, fac in cfg["phi_factors"].items():
                    f.write("{0} {1}\n".format(tag, fac))
    obs.loc[["fc_shared", "fc_struct"], "obsval"] = outs_t[cfg["n_obs"]:]
    obs.loc[["fc_shared", "fc_struct"], "weight"] = 0.0
    obs.loc[["fc_shared", "fc_struct"], "obgnme"] = "fc"
    # the shared prior and noise realizations: every arm of a setting starts from these
    rng = np.random.default_rng(cfg["ens_seed"])
    rnames = [str(i) for i in range(cfg["n_reals"])]
    pe = pd.DataFrame(cfg["prior_sd"] * rng.standard_normal((cfg["n_reals"], cfg["k"])),
                      index=rnames, columns=pn)
    pe.index.name = "real_name"
    pe.to_csv(os.path.join(t_d, "prior.csv"))
    oe = pd.DataFrame(np.tile(obs.loc[on, "obsval"].values, (cfg["n_reals"], 1)), index=rnames, columns=on)
    oe.loc[:, rec] += cfg["sigma"] * rng.standard_normal((cfg["n_reals"], cfg["n_obs"]))
    oe.index.name = "real_name"
    oe.to_csv(os.path.join(t_d, "noise.csv"))
    sched = [cfg["first_cycle"]] + [cfg["cycle"]] * (cfg["n_cycles"] - 2) + [999]
    pst.control_data.noptmax = cfg["first_cycle"] + cfg["cycle"] * (cfg["n_cycles"] - 1)
    pst.model_command = ["{0} forward_run.py".format(sys.executable)]
    pst.pestpp_options = {
        "ies_par_en": "prior.csv", "ies_obs_en": "noise.csv", "ies_num_reals": cfg["n_reals"],
        "ies_include_base": False, "ies_no_noise": False, "random_seed": 4321,
        "ies_n_iter_reinflate": ",".join(str(s) for s in sched),
        "ies_bad_phi_sigma": 1.0e30, "overdue_giveup_fac": 1.0e30,
    }
    if cfg.get("phi_factors_by_real"):
        pst.pestpp_options["ies_phi_factor_file"] = "phi_factors.csv"
        pst.pestpp_options["ies_phi_factors_by_real"] = True
    elif cfg.get("phi_factors"):
        pst.pestpp_options["ies_phi_factor_file"] = "phi_factors.dat"
    pst.write(os.path.join(t_d, "case.pst"), version=2)
    return pst


def run_one(job):
    """run one setting x arm in its own copy of the setting's template"""
    import pyemu
    t_d, m_d, opts, exe = job
    if os.path.exists(os.path.join(m_d, "case.phi.actual.csv")):
        return m_d, "skipped"
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(t_d, m_d)
    pst = pyemu.Pst(os.path.join(m_d, "case.pst"))
    pst.pestpp_options.update(opts)
    pst.write(os.path.join(m_d, "case.pst"), version=2)
    try:
        pyemu.os_utils.run("{0} case.pst /e".format(exe), cwd=m_d)
    except Exception as e:
        return m_d, "failed: {0}".format(e)
    return m_d, "ok"


# ---------------------------------------------------------------- scoring

def capture_files(m_d):
    """the adjusted control file weights pestpp-ies saves at each reinflation that changes them,
    in iteration order"""
    fs = glob.glob(os.path.join(m_d, "case.*.adjusted.obs_data.csv"))
    fs = [f for f in fs if int(os.path.basename(f).split(".")[1]) > 0]
    return sorted(fs, key=lambda f: int(os.path.basename(f).split(".")[1]))


def last_fitted_iteration(m_d):
    """the last iteration that was a fit, not a reinflation.  pestpp-ies counts each
    reinflation as an iteration and marks it with case.N.reinflate.pcs.csv; with
    ies_n_iter_reinflate(1) the run ends on a reinflation, whose ensemble is not a posterior"""
    its = pd.read_csv(os.path.join(m_d, "case.phi.actual.csv")).iteration.astype(int).tolist()
    fitted = [i for i in its if not os.path.exists(os.path.join(m_d, "case.{0}.reinflate.pcs.csv".format(i)))]
    return max(fitted)


def base_weights(cfg, m_d):
    """the weights the run started from: the user's, or with phi factors the rebalanced ones
    pestpp-ies saved as case.adjusted.obs_data.csv"""
    n = cfg["n_obs"]
    f = os.path.join(m_d, "case.adjusted.obs_data.csv")
    if cfg.get("phi_factors") and os.path.exists(f):
        df = pd.read_csv(f, index_col=0)
        df.index = df.index.str.lower()
        return df.loc[obs_names(cfg)[:n], "weight"].values.astype(float)
    return np.full(n, 1.0 / (cfg["weight_sd_factor"] * cfg["sigma"]))


def score(cfg, m_d, arm, arm_opts, th):
    """rows per capture (the added variance against the known answers) and one final row"""
    n = cfg["n_obs"]
    rec = obs_names(cfg)[:n]
    w0 = base_weights(cfg, m_d)
    if cfg.get("phi_factors"):
        th = theory(cfg, weights=w0)
    frac = arm_frac(arm_opts)
    em = arm_opts.get("ies_structnoise_em", False)
    rows, added_last = [], np.zeros(n)
    for icap, f in enumerate(capture_files(m_d)):
        df = pd.read_csv(f, index_col=0)
        df.index = df.index.str.lower()
        w = df.loc[rec, "weight"].values.astype(float)
        added = 1.0 / w ** 2 - 1.0 / w0 ** 2
        added_last = added
        pred_first = frac * (th["excess_first"] if em else th["s2_first"])
        rows.append({
            "capture": icap + 1, "iteration": int(os.path.basename(f).split(".")[1]),
            "added_mean": added.mean(), "s2_struct_mean": th["s2_struct"].mean(),
            "added_over_struct": added.mean() / max(th["s2_struct"].mean(), 1e-30),
            "added_over_sig2": added.mean() / th["sig2"].mean(),
            "frac_obs_corrected": float((added > 1e-12).mean()),
            "corr_with_struct": float(np.corrcoef(added, th["s2_struct"])[0, 1])
            if (added.std() > 0 and th["s2_struct"].std() > 0) else np.nan,
            "first_capture_theory_mean": pred_first.mean() if icap == 0 else np.nan,
            "first_capture_over_theory": added.mean() / pred_first.mean()
            if (icap == 0 and pred_first.mean() > 0) else np.nan,
        })
    phi = pd.read_csv(os.path.join(m_d, "case.phi.actual.csv"))
    last = last_fitted_iteration(m_d)
    pe = pd.read_csv(os.path.join(m_d, "case.{0}.par.csv".format(last)), index_col=0)
    pe.columns = pe.columns.str.lower()
    oe = pd.read_csv(os.path.join(m_d, "case.{0}.obs.csv".format(last)), index_col=0)
    oe.columns = oe.columns.str.lower()
    theta_t, outs_t, _, _ = truth(cfg)
    pv = pe.loc[:, par_names(cfg)].values
    lo, hi = np.percentile(pv, 5, axis=0), np.percentile(pv, 95, axis=0)
    fin = {"final_iteration": last, "final_phi_actual_mean": float(phi.loc[phi.iteration == last, "mean"].iloc[0]),
           "base_weight_min": float(w0.min()), "base_weight_max": float(w0.max()),
           "par_bias": float(np.mean(np.abs(pv.mean(axis=0) - theta_t)) / cfg["prior_sd"]),
           "par_cover90": float(np.mean((theta_t >= lo) & (theta_t <= hi))),
           "par_sd_mean": float(pv.std(axis=0).mean())}
    for j, fc in enumerate(["fc_shared", "fc_struct"]):
        v = oe.loc[:, fc].values
        tv = outs_t[n + j]
        fin[fc + "_err"] = float(v.mean() - tv)
        fin[fc + "_sd"] = float(v.std())
        fin[fc + "_cover90"] = bool(np.percentile(v, 5) <= tv <= np.percentile(v, 95))
    return rows, fin, added_last, oe


def report(out_d, sets, the_arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = arm_colors()
    cap_rows, fin_rows = [], []
    pdf = PdfPages(os.path.join(out_d, "structnoise_stress.pdf"))
    for sname, over in sets.items():
        cfg = dict(base_config(), **over)
        th = theory(cfg)
        t = times(cfg)
        theta_t, outs_t, d_obs, g = truth(cfg)
        res = {}
        for aname, aopts in the_arms.items():
            m_d = os.path.join(out_d, "runs", "{0}__{1}".format(sname, aname))
            if not os.path.exists(os.path.join(m_d, "case.phi.actual.csv")):
                continue
            rows, fin, added_last, oe = score(cfg, m_d, aname, aopts, th)
            for r in rows:
                cap_rows.append(dict({"setting": sname, "arm": aname, "p_eff": th["p_eff"],
                                      "n_obs": cfg["n_obs"]}, **r))
            fin_rows.append(dict({"setting": sname, "arm": aname, "p_eff": th["p_eff"],
                                  "s2_struct_mean": th["s2_struct"].mean(),
                                  "sig2": cfg["sigma"] ** 2}, **fin))
            res[aname] = (rows, added_last, oe)
        if len(res) == 0:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
        fig.suptitle("{0}: {1}   (p_eff {2:.2f}, n {3})".format(
            sname, ", ".join("{0}={1}".format(k, v) for k, v in over.items()) or "base",
            th["p_eff"], cfg["n_obs"]), fontsize=10)
        rec = obs_names(cfg)[:cfg["n_obs"]]
        for ax, aname in zip(axes[0], ["none", "em_recap"]):
            if aname not in res:
                continue
            oe = res[aname][2]
            ax.plot(t, oe.loc[:, rec].values.T, color=colors[aname], lw=0.4, alpha=0.4)
            ax.plot(t, outs_t[:cfg["n_obs"]], color="black", lw=1.2, label="truth")
            ax.plot(t, d_obs, "o", color="#c53030", ms=2.5, label="observed")
            ax.set_title("final realizations: " + aname, fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
        ax = axes[1, 0]
        ax.plot(t, th["s2_struct"], color="black", lw=2.0, label="known structure (s2_struct)")
        ax.axhline(cfg["sigma"] ** 2, color="black", ls="--", lw=1.0, label="noise variance")
        for aname, (rows, added_last, _) in res.items():
            if aname == "none":
                continue
            ax.plot(t, added_last, color=colors[aname], lw=1.0, label=aname)
        ax.set_xlabel("time")
        ax.set_ylabel("added variance at the last capture")
        ax.legend(fontsize=6)
        ax = axes[1, 1]
        for aname, (rows, _, _) in res.items():
            if aname == "none" or len(rows) == 0:
                continue
            ref = "added_over_struct" if th["s2_struct"].mean() > 1e-12 else "added_over_sig2"
            ax.plot([r["capture"] for r in rows], [r[ref] for r in rows], "o-", color=colors[aname],
                    label=aname)
        if th["s2_struct"].mean() > 1e-12:
            ax.axhline(1.0, color="black", lw=1.0)
            ax.axhline(1.0 - th["p_eff"] / cfg["n_obs"], color="black", ls=":", lw=1.0)
            ax.set_ylabel("mean added variance / mean known structure")
        else:
            ax.set_ylabel("mean added variance / noise variance")
        ax.set_xlabel("capture")
        ax.legend(fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    caps = pd.DataFrame(cap_rows)
    fins = pd.DataFrame(fin_rows)
    caps.to_csv(os.path.join(out_d, "summary.csv"), index=False)
    fins.to_csv(os.path.join(out_d, "final.csv"), index=False)
    return caps, fins


def default_exe():
    here = os.path.dirname(os.path.abspath(__file__))
    plat = {"darwin": "mac", "win32": "win"}.get(sys.platform, "linux")
    name = "pestpp-ies.exe" if plat == "win" else "pestpp-ies"
    return os.path.abspath(os.path.join(here, "..", "bin", plat, name))


def run_setting(sname, arm_names, out_d, num_workers=4, exe=None, fresh=True):
    """run some arms of one setting and score them.  returns (capture rows, final rows) as
    dataframes, the same columns as summary.csv and final.csv.  used by the ci tests"""
    exe = exe or default_exe()
    over = settings()[sname]
    cfg = dict(base_config(), **over)
    the_arms = {a: arms()[a] for a in arm_names}
    runs = os.path.join(out_d, "runs")
    os.makedirs(runs, exist_ok=True)
    t_d = os.path.join(runs, "template_" + sname)
    build_template(cfg, t_d)
    jobs = []
    for aname, aopts in the_arms.items():
        m_d = os.path.join(runs, "{0}__{1}".format(sname, aname))
        if fresh and os.path.exists(m_d):
            shutil.rmtree(m_d)
        jobs.append((t_d, m_d, aopts, exe))
    with mp.Pool(min(num_workers, len(jobs))) as pool:
        status = pool.map(run_one, jobs)
    bad = [s for s in status if s[1] not in ("ok", "skipped")]
    if len(bad) > 0:
        raise Exception("runs failed: {0}".format(bad))
    th = theory(cfg)
    cap_rows, fin_rows = [], []
    for aname, aopts in the_arms.items():
        m_d = os.path.join(runs, "{0}__{1}".format(sname, aname))
        rows, fin, _, _ = score(cfg, m_d, aname, aopts, th)
        cap_rows.extend(dict({"setting": sname, "arm": aname, "p_eff": th["p_eff"],
                              "n_obs": cfg["n_obs"]}, **r) for r in rows)
        fin_rows.append(dict({"setting": sname, "arm": aname, "p_eff": th["p_eff"],
                              "sig2": cfg["sigma"] ** 2}, **fin))
    return pd.DataFrame(cap_rows), pd.DataFrame(fin_rows)


def main(quick=False, report_only=False, num_workers=8, out_d=None,
         exe=None):
    here = os.path.dirname(os.path.abspath(__file__))
    out_d = out_d or os.path.join(here, "structnoise_stress")
    exe = exe or default_exe()
    sets = settings()
    if quick:
        sets = {"base": sets["base"]}
    the_arms = arms()
    if not report_only:
        os.makedirs(os.path.join(out_d, "runs"), exist_ok=True)
        jobs = []
        for sname, over in sets.items():
            cfg = dict(base_config(), **over)
            t_d = os.path.join(out_d, "runs", "template_" + sname)
            build_template(cfg, t_d)
            for aname, aopts in the_arms.items():
                jobs.append((t_d, os.path.join(out_d, "runs", "{0}__{1}".format(sname, aname)), aopts, exe))
        with mp.Pool(num_workers) as pool:
            for m_d, status in pool.imap_unordered(run_one, jobs):
                print(os.path.basename(m_d), status, flush=True)
    caps, fins = report(out_d, sets, the_arms)
    pd.set_option("display.width", 200)
    print(caps.loc[caps.capture == caps.groupby(["setting", "arm"]).capture.transform("max"),
                   ["setting", "arm", "capture", "added_over_struct", "added_over_sig2",
                    "first_capture_over_theory"]].round(3).to_string(index=False))
    print(fins[["setting", "arm", "final_phi_actual_mean", "par_bias", "par_cover90",
                "fc_shared_cover90", "fc_struct_cover90"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(quick=a.quick, report_only=a.report, num_workers=a.workers)
