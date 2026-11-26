"""helpers for the run manager overdue tests in basic_tests.py: the hanging
analytical model, the per tool control files, the rmr parsing and the checks.
not a test module itself (nothing in here ends in _test).

the model has four parameters and a handful of outputs that serve as
observations, objectives and constraints.  it hangs (sleeps forever) either
on every hang_every-th run of a worker, staggered by worker so the workers do
not all hang in the same batch, or on one exact run number counted across
all workers, or whenever p0 is below a threshold (a hang that belongs to the
parameter set and so comes back on every retry).  with max_run_fail 1 a hung
run is given up on once it crosses the overdue threshold, recorded as timed
out and dropped; with a higher max_run_fail it is requeued.

the panther master logs every giveup in the rmr file as
    ->sending kill request. reason:overdue. duration:D, avg:A ...
and every batch start as
    running model N times
so the tests can say in which batch each kill happened and what drove it.

these need panther workers, so local port binding has to be allowed."""
import os
import re
import sys
import time
import shutil
import platform
import subprocess
import numpy as np
import pandas as pd
import pyemu


def exe_for(tool):
    """absolute path to a tool binary.  the worker dirs sit deeper than the
    convention the relative benchmark paths assume, so make it absolute"""
    if "windows" in platform.platform().lower():
        plat, ext = "win", ".exe"
    elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
        plat, ext = "mac", ""
    else:
        plat, ext = "linux", ""
    cands = [os.path.join("..", "bin", plat, "pestpp-" + tool + ext),
             os.path.join("..", "..", "pestpp", "bin", plat, "pestpp-" + tool + ext),
             os.path.join("..", "..", "..", "..", "pestpp", "bin", plat, "pestpp-" + tool + ext)]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(cands[0])


def forward_run(ws="."):
    """the model.  overdue_cfg.txt holds: base_sleep seconds, hang_every,
    and optionally a global run number to hang on and a p0 threshold below
    which the run hangs"""
    p = np.loadtxt(os.path.join(ws, "par.dat"))
    base_sleep, hang_every, hang_at, hang_below = 0.0, 0, 0, -1.0
    cf = os.path.join(ws, "overdue_cfg.txt")
    if os.path.exists(cf):
        parts = open(cf).read().split()
        base_sleep, hang_every = float(parts[0]), int(parts[1])
        if len(parts) > 2:
            hang_at = int(parts[2])
        if len(parts) > 3:
            hang_below = float(parts[3])
    if base_sleep > 0:
        time.sleep(base_sleep)
    hang = False
    if hang_below > 0.0 and p[0] < hang_below:
        hang = True
    elif hang_at > 0:
        # one hang at an exact run number counted across every worker.  the counter
        # sits one level up, in the worker root, behind a mkdir lock
        root = os.path.dirname(os.path.abspath(ws))
        lock = os.path.join(root, "overdue_counter.lock")
        for _ in range(2000):
            try:
                os.mkdir(lock)
                break
            except OSError:
                time.sleep(0.01)
        gf = os.path.join(root, "overdue_global_counter.txt")
        n = int(open(gf).read()) + 1 if os.path.exists(gf) else 1
        with open(gf, "w") as f:
            f.write(str(n))
        try:
            os.rmdir(lock)
        except OSError:
            pass
        hang = (n == hang_at)
    elif hang_every > 0:
        cnt_f = os.path.join(ws, "overdue_counter.txt")
        n = int(open(cnt_f).read()) + 1 if os.path.exists(cnt_f) else 1
        with open(cnt_f, "w") as f:
            f.write(str(n))
        # stagger by worker: the workers all do the same number of runs per batch, so
        # without an offset they would all hang in the same batch
        try:
            offset = int(os.path.basename(os.path.abspath(ws)).split("_")[-1])
        except ValueError:
            offset = 0
        hang = (n + offset) % hang_every == 0
    if hang:
        with open(os.path.join(ws, "overdue_hangs.txt"), "a") as f:
            f.write("{0} {1}\n".format(p[0], time.time()))
        time.sleep(1.0e7)
    g = 1.0 + 9.0 * np.mean(p[1:])
    out = {"lin0": p[0], "lin1": p[1], "lin2": p[2], "lin3": p[3],
           "sumsq": float(np.sum((p - 0.5) ** 2)),
           "f1": p[0], "f2": g * (1.0 - np.sqrt(max(p[0], 0.0) / g)),
           "constr": p[0] + p[1]}
    with open(os.path.join(ws, "obs.dat"), "w") as f:
        for k, v in out.items():
            f.write("{0} {1:25.16E}\n".format(k, v))


def write_cfg(template_d, base_sleep, hang_every, hang_at=0, hang_below=-1.0):
    with open(os.path.join(template_d, "overdue_cfg.txt"), "w") as f:
        f.write("{0} {1} {2} {3}\n".format(base_sleep, hang_every, hang_at, hang_below))


def write_template(template_d, base_sleep, hang_every):
    if os.path.exists(template_d):
        shutil.rmtree(template_d)
    os.makedirs(template_d)
    with open(os.path.join(template_d, "par.dat.tpl"), "w") as f:
        f.write("ptf ~\n")
        for k in range(4):
            f.write("~  p{0}  ~\n".format(k))
    np.savetxt(os.path.join(template_d, "par.dat"), np.full(4, 0.5))
    write_cfg(template_d, base_sleep, hang_every)
    forward_run(template_d)
    for f in ("overdue_counter.txt",):
        if os.path.exists(os.path.join(template_d, f)):
            os.remove(os.path.join(template_d, f))
    onames = [l.split()[0] for l in open(os.path.join(template_d, "obs.dat"))]
    with open(os.path.join(template_d, "obs.dat.ins"), "w") as f:
        f.write("pif ~\n")
        for n in onames:
            f.write("l1 w !{0}!\n".format(n))
    shutil.copy2(os.path.abspath(__file__), os.path.join(template_d, "overdue_utils.py"))
    with open(os.path.join(template_d, "forward_run.py"), "w") as f:
        f.write("import overdue_utils\nif __name__ == '__main__':\n    overdue_utils.forward_run()\n")
    pst = pyemu.Pst.from_io_files([os.path.join(template_d, "par.dat.tpl")], [os.path.join(template_d, "par.dat")],
                                  [os.path.join(template_d, "obs.dat.ins")], [os.path.join(template_d, "obs.dat")],
                                  pst_path=".")
    par = pst.parameter_data
    par.loc[:, "partrans"] = "none"
    par.loc[:, "parchglim"] = "relative"
    par.loc[:, "parval1"] = 0.5
    par.loc[:, "parlbnd"] = 0.0
    par.loc[:, "parubnd"] = 1.0
    par.loc[:, "pargp"] = "dv"
    pst.rectify_pgroups()
    obs = pst.observation_data
    obs.loc[:, "weight"] = 0.0
    obs.loc[:, "obgnme"] = "obs"
    obs.loc[["lin0", "lin1", "lin2", "lin3"], "weight"] = 1.0
    obs.loc[["lin0", "lin1", "lin2", "lin3"], "obsval"] = [0.2, 0.4, 0.6, 0.8]
    pst.model_command = "python forward_run.py"
    pst.control_data.noptmax = 0
    return pst


def overdue_options(variant):
    """the two ways of driving the giveup, and the model sleep that goes with
    each.  the factor variant needs the average above the 0.01 minute floor
    the master applies before it will use a factor"""
    if variant == "minutes":
        return {"overdue_giveup_minutes": 0.25, "overdue_giveup_fac": 1.0e10}, 0.3
    return {"overdue_giveup_minutes": 1.0e10, "overdue_giveup_fac": 3.0}, 2.0


def tool_setup(tool, pst, template_d):
    """per tool control settings on top of the shared problem.  returns the
    case name and the number of workers"""
    opts = pst.pestpp_options
    if tool == "ies":
        pst.control_data.noptmax = 3
        opts.update({"ies_num_reals": 20, "ies_include_base": "false", "ies_lambda_mults": "1.0",
                     "lambda_scale_fac": "1.0", "ies_subset_size": 4, "ies_bad_phi_sigma": 1.0e10})
    elif tool == "glm":
        pst.control_data.noptmax = 3
        pst.parameter_groups.loc["dv", "derinc"] = 0.05
        pst.parameter_groups.loc["dv", "inctyp"] = "absolute"
        # eight upgrade runs per iteration rather than two: a batch that small can lose
        # every run to a hang, and glm stops when all its upgrade runs fail
        opts.update({"lambdas": "0.1,1.0,10.0,100.0", "lambda_scale_fac": "0.5,1.0", "uncertainty": "false"})
    elif tool == "mou":
        pst.control_data.noptmax = 3
        opts.update({"mou_objectives": "f1,f2", "mou_population_size": 16, "opt_dec_var_groups": "dv",
                     "mou_generator": "de", "mou_env_selector": "nsga"})
        # objectives must carry weight, and sit in a less-than group to be minimized
        pst.observation_data.loc[["f1", "f2"], "obgnme"] = "less_than_obj"
        pst.observation_data.loc[["f1", "f2"], "weight"] = 1.0
    elif tool == "da":
        pst.control_data.noptmax = 2
        # one instruction file serves every cycle, so every observation lives in every
        # cycle and the weight table decides which ones count when
        obs = pst.observation_data
        obs.loc[:, "cycle"] = -1
        obs.loc[["lin0", "lin1", "lin2", "lin3"], "weight"] = 1.0
        wt = pd.DataFrame(0.0, index=["lin0", "lin1", "lin2", "lin3"], columns=[0, 1])
        wt.loc[["lin0", "lin1"], 0] = 1.0
        wt.loc[["lin2", "lin3"], 1] = 1.0
        wt.index.name = "obsnme"
        wt.to_csv(os.path.join(template_d, "weight_cycle_tbl.csv"))
        opts["da_weight_cycle_table"] = "weight_cycle_tbl.csv"
        pst.parameter_data.loc[:, "cycle"] = -1
        pst.model_input_data.loc[:, "cycle"] = -1
        pst.model_output_data.loc[:, "cycle"] = -1
        opts.update({"da_num_reals": 16, "ies_include_base": "false", "ies_lambda_mults": "1.0",
                     "lambda_scale_fac": "1.0", "ies_subset_size": 4, "ies_bad_phi_sigma": 1.0e10})
    elif tool == "sqp":
        pst.control_data.noptmax = 3
        # sqp applies the phi based stopping rules; switch them off so it runs every iteration,
        # and start away from the optimum (the objective is zero at 0.5) or it stops at once
        pst.control_data.phiredstp = 0.0
        pst.control_data.nphistp = 100
        pst.control_data.nphinored = 100
        pst.parameter_data.loc[:, "parval1"] = 0.2
        opts.update({"opt_obj_func": "sumsq", "opt_dec_var_groups": "dv", "opt_direction": "min",
                     "opt_constraint_groups": "l_constr", "sqp_num_reals": 16, "sqp_alpha_mults": "1.0"})
        # the objective observation needs weight but must not be in a constraint group,
        # and there has to be at least one constraint
        pst.observation_data.loc["sumsq", "obgnme"] = "obj"
        pst.observation_data.loc["sumsq", "weight"] = 1.0
        pst.observation_data.loc["constr", "obgnme"] = "l_constr"
        pst.observation_data.loc["constr", "obsval"] = 1.5
        pst.observation_data.loc["constr", "weight"] = 1.0
    elif tool == "opt":
        pst.control_data.noptmax = 2
        pst.add_pi_equation(["p0", "p1", "p2", "p3"], coef_dict={"p0": 1.0, "p1": 1.0, "p2": 1.0, "p3": 1.0},
                            pilbl="obj", obs_group="obj")
        pst.observation_data.loc["constr", "obgnme"] = "l_constr"
        pst.observation_data.loc["constr", "obsval"] = 1.5
        pst.observation_data.loc["constr", "weight"] = 1.0
        opts.update({"opt_obj_func": "obj", "opt_dec_var_groups": "dv", "opt_direction": "max",
                     "opt_constraint_groups": "l_constr", "opt_risk": 0.5})
        # pestpp-opt cannot fill a response matrix with a run missing, so one hang, pinned to
        # run 8 counted across the workers: runs 1-5 are the first response matrix, 6 the
        # best-decision-variable run, 7-10 the second response matrix
        base_sleep = open(os.path.join(template_d, "overdue_cfg.txt")).read().split()[0]
        write_cfg(template_d, base_sleep, 0, hang_at=8)
    elif tool == "swp":
        pst.control_data.noptmax = 0
        n = 40
        df = pd.DataFrame(np.random.default_rng(3).uniform(0.05, 0.95, (n, 4)), columns=pst.par_names,
                          index=["sweep_{0}".format(i) for i in range(n)])
        df.to_csv(os.path.join(template_d, "sweep_in.csv"))
        opts.update({"sweep_parameter_csv_file": "sweep_in.csv"})
    elif tool == "sen":
        pst.control_data.noptmax = 0
        opts.update({"gsa_method": "morris", "gsa_morris_r": 6, "gsa_morris_p": 4})
    else:
        raise ValueError(tool)
    return "overdue", 4


def parse_rmr(rmr_file):
    """kill events with their batch index, duration and average, from the rmr"""
    batch = -1
    kills = []
    for line in open(rmr_file):
        if line.startswith("running model"):
            batch += 1
        m = re.search(r"reason:overdue\. duration:([0-9.eE+-]+), avg:([0-9.eE+-]+)", line)
        if m:
            kills.append({"batch": batch, "duration": float(m.group(1)), "avg": float(m.group(2))})
    return pd.DataFrame(kills, columns=["batch", "duration", "avg"]), batch + 1


def finished(tool, m_d, case, noptmax):
    """did the tool get through what it was asked for"""
    if tool == "ies":
        df = pd.read_csv(os.path.join(m_d, case + ".phi.actual.csv"))
        return int(df["iteration"].values[-1]) == noptmax
    if tool == "da":
        f = os.path.join(m_d, case + ".global.phi.actual.csv")
        if not os.path.exists(f):
            return False
        df = pd.read_csv(f)
        return df["cycle"].nunique() == 2 and int(df["iteration"].max()) == noptmax
    if tool == "glm":
        return os.path.exists(os.path.join(m_d, case + ".iobj")) and \
            pd.read_csv(os.path.join(m_d, case + ".iobj")).shape[0] >= noptmax
    if tool == "mou":
        return os.path.exists(os.path.join(m_d, "{0}.{1}.dv_pop.csv".format(case, noptmax)))
    if tool == "sqp":
        return os.path.exists(os.path.join(m_d, "{0}.{1}.par.csv".format(case, noptmax)))
    if tool == "opt":
        # pestpp-opt cannot fill a response matrix with a run missing, so a giveup during
        # the matrix is the end of the run by design.  "finished" here means opt got past
        # its first iteration and said why it stopped
        rec = open(os.path.join(m_d, case + ".rec")).read().lower()
        return ("starting lp iteration 2" in rec) and \
            (("cannot continue" in rec) or ("reason for terminating" in rec))
    if tool == "swp":
        return os.path.exists(os.path.join(m_d, "sweep_out.csv"))
    if tool == "sen":
        return os.path.exists(os.path.join(m_d, case + ".msn")) or os.path.exists(os.path.join(m_d, case + ".mio"))
    return False


def hang_every_for(tool):
    return {"glm": 4}.get(tool, 5)


def run_tool(tool, variant, port, model_d="run_mgr_overdue", exe=None, extra=None, watchdog_s=None):
    """build and run one case.  returns (kills dataframe, number of batches,
    finished flag, master error or None).  with watchdog_s the master and the
    workers are started directly and killed if the master is not done in time,
    which is how a hang in the run manager itself shows up as a failure rather
    than a stuck test"""
    os.makedirs(model_d, exist_ok=True)
    opts, base_sleep = overdue_options(variant)
    if extra is not None:
        opts = dict(opts)
        opts.update(extra)
    template_d = os.path.join(model_d, "template_{0}_{1}".format(tool, variant))
    pst = write_template(template_d, base_sleep, hang_every_for(tool))
    case, num_workers = tool_setup(tool, pst, template_d)
    pst.pestpp_options.update(opts)
    pst.pestpp_options.setdefault("max_run_fail", 1)
    pst.pestpp_options["panther_agent_freeze_on_fail"] = "false"
    pst.pestpp_options["panther_ping_interval_secs"] = 1
    pst.write(os.path.join(template_d, case + ".pst"), version=2)
    noptmax = pst.control_data.noptmax
    m_d = os.path.join(model_d, "master_{0}_{1}".format(tool, variant))
    # workers get their own root per case: two runs sharing worker_0..3 wipe each other out
    w_root = os.path.join(model_d, "workers_{0}_{1}".format(tool, variant))
    if os.path.exists(w_root):
        shutil.rmtree(w_root)
    os.makedirs(w_root)
    if exe is None:
        exe = exe_for(tool)
    master_error = None
    if watchdog_s is None:
        try:
            pyemu.os_utils.start_workers(template_d, exe, case + ".pst", num_workers=num_workers,
                                         worker_root=w_root, master_dir=m_d, port=port)
        except Exception as e:
            master_error = str(e)
    else:
        master_error = _run_with_watchdog(template_d, m_d, w_root, exe, case, num_workers, port, watchdog_s)
    shutil.rmtree(w_root, ignore_errors=True)
    rmr = os.path.join(m_d, case + ".rmr")
    kills, nbatch = parse_rmr(rmr) if os.path.exists(rmr) else (pd.DataFrame(columns=["batch", "duration", "avg"]), 0)
    done = finished(tool, m_d, case, noptmax) if os.path.exists(os.path.join(m_d, case + ".rec")) else False
    if master_error is not None:
        print("{0}/{1}: master exited with an error: {2}".format(tool, variant, master_error))
        # pestpp-opt stops on a given up response matrix run by design, and as of this
        # writing leaves on a signal rather than a clean error code; finished() judges it
        if tool != "opt":
            done = False
    return kills, nbatch, done, master_error


def _run_with_watchdog(template_d, m_d, w_root, exe, case, num_workers, port, watchdog_s):
    """master and workers as plain subprocesses, everything killed if the
    master outlives the watchdog.  returns an error string or None"""
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    shutil.copytree(template_d, m_d)
    procs = []
    with open(os.path.join(m_d, "master_stdout.txt"), "w") as mlog:
        master = subprocess.Popen([exe, case + ".pst", "/h", ":{0}".format(port)], cwd=m_d, stdout=mlog,
                                  stderr=subprocess.STDOUT)
        for k in range(num_workers):
            wd = os.path.join(w_root, "worker_{0}".format(k))
            shutil.copytree(template_d, wd)
            wlog = open(os.path.join(wd, "worker_stdout.txt"), "w")
            procs.append((subprocess.Popen([exe, case + ".pst", "/h", "localhost:{0}".format(port)], cwd=wd,
                                           stdout=wlog, stderr=subprocess.STDOUT), wlog))
        t0 = time.time()
        timed_out = False
        while master.poll() is None:
            if time.time() - t0 > watchdog_s:
                timed_out = True
                master.kill()
                break
            time.sleep(1.0)
        for p, wlog in procs:
            try:
                p.kill()
            except Exception:
                pass
            wlog.close()
    if timed_out:
        return "watchdog: master still running after {0} s".format(watchdog_s)
    if master.returncode != 0:
        return "master returned non-zero: {0}".format(master.returncode)
    return None


def check(tool, variant, kills, nbatch, done, single_batch=False):
    """the assertions for one case, returned as a list of problems"""
    msg = "{0}/{1}: ".format(tool, variant)
    print("\n" + msg + "{0} batches, {1} overdue kills in batches {2}, finished: {3}".format(
        nbatch, kills.shape[0], sorted(kills.batch.unique().tolist()) if kills.shape[0] else [], done))
    if kills.shape[0]:
        print(kills.groupby("batch").agg(n=("duration", "size"), dur_min=("duration", "min"),
                                         dur_max=("duration", "max"), avg=("avg", "mean")).round(3).to_string())
    errs = []
    if not done:
        errs.append(msg + ("opt did not reach its second iteration before the giveup ended it, or did not say why it stopped"
                           if tool == "opt" else "the tool did not finish its iterations"))
    if kills.shape[0] == 0:
        errs.append(msg + "no overdue kills recorded - the hang never tripped the giveup")
        return errs
    if tool == "opt":
        # the first giveup ends an opt run, so one batch is all there can be; what matters
        # is that it happened in a later iteration (batch 0 is the first response matrix,
        # batch 1 the best-decision-variable run, batch 2 the second response matrix)
        if kills.batch.max() < 2:
            errs.append(msg + "the giveup happened in the first iteration, later iterations were not exercised")
    elif not single_batch and kills.batch.nunique() < 2:
        errs.append(msg + "kills in only one batch, so later iterations were not exercised")
    if variant == "minutes":
        bad = kills.loc[(kills.duration < 0.25) | (kills.duration > 0.6)]
        if bad.shape[0]:
            errs.append(msg + "giveup_minutes 0.25 did not set the kill time: durations {0}".format(
                bad.duration.round(3).tolist()))
    else:
        # the rec prints both numbers rounded, so allow a couple of percent under the factor
        bad = kills.loc[(kills.duration < 0.98 * 3.0 * kills.avg) | (kills.duration > 1.0)]
        if bad.shape[0]:
            errs.append(msg + "giveup_fac 3 did not set the kill time: duration/avg {0}".format(
                (bad.duration / bad.avg).round(2).tolist()))
    return errs


def run_set(tools, variant, port_base, single=(), exe=None):
    """run a group of tools under one variant and assert on all of them at once"""
    errs = []
    port = port_base
    for tool in tools:
        kills, nbatch, done, _ = run_tool(tool, variant, port, exe=exe)
        errs += check(tool, variant, kills, nbatch, done, single_batch=(tool in single))
        port += 1
    assert not errs, "\n".join(errs)
