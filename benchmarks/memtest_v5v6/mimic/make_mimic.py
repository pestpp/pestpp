"""a scaled-down copy of the ruataniwha noptmax=-2 setup, for comparing master memory
between pestpp-ies versions.

copies the things that matter for memory, at 'scale' of the real thing (1.0 = full size):
  - 7.54M obs in 3410 groups, 1.5% weighted, ~85 character names
  - a 34 column external obs data csv, mostly empty cells, with standard_deviation on 64k
    obs and lower_bound on 997k
  - 47k parameters in a 27 column external par data csv, prior ensemble as a binary file
  - ~30 model output files read through instruction files
  - the same '++' options, including the phi factor file, 50 reals and a serial run
"""
import os
import sys
import numpy as np
import pandas as pd
import pyemu


def main(scale=0.1, num_reals=50, n_prior_reals=200, n_out_files=30):
    d = os.path.dirname(os.path.abspath(__file__))
    os.chdir(d)
    rng = np.random.default_rng(13)

    nobs = int(7543962 * scale)
    npar = int(47038 * scale)
    ngroups = max(int(3410 * scale), 20)
    n_nz = int(110877 * scale)
    n_std = int(64436 * scale)
    n_lb = int(997003 * scale)

    # parameters
    pnames = ["pname:hkgr_inst:0_ptype:gr_usecol:strhc1_fac_pstyle:m_i:{0}_j:{1}_reachid:{2}_iseg:0".format(
        i // 100, i % 100, i) for i in range(npar)]
    with open("model.in.tpl", "w") as f:
        f.write("ptf ~\n")
        for p in pnames:
            f.write("~ {0:<120s}~\n".format(p))
    pdf = pd.DataFrame({"parnme": pnames, "partrans": "log", "parchglim": "factor", "parval1": 1.0,
                        "parlbnd": 0.2, "parubnd": 5.0, "pargp": ["hkgr_k:{0}".format(i % 3) for i in range(npar)],
                        "scale": 1.0, "offset": 0.0, "dercom": 1})
    for c in ["i", "idx0", "idx1", "idx2", "inst", "iseg", "j", "k", "month", "pname", "pstyle", "ptype",
              "reachid", "usecol", "x", "y", "zone"]:
        pdf[c] = ""
    pdf["i"] = [i // 100 for i in range(npar)]
    pdf["j"] = [i % 100 for i in range(npar)]
    pdf["pname"] = "hkgr"
    pdf["ptype"] = "gr"
    pdf.to_csv("mimic.par_data.csv", index=False)
    gdf = pd.DataFrame({"pargpnme": ["hkgr_k:{0}".format(i) for i in range(3)], "inctyp": "relative",
                        "derinc": 0.01, "derinclb": 0.0, "forcen": "switch", "derincmul": 2.0,
                        "dermthd": "parabolic", "splitthresh": 1e-5, "splitreldiff": 0.5, "splitaction": "smaller"})
    gdf.to_csv("mimic.pargp_data.csv", index=False)
    pd.DataFrame({"pest_file": ["model.in.tpl"], "model_file": ["model.in"]}).to_csv(
        "mimic.tplfile_data.csv", index=False)

    # observations: group names carry the phi factor tags, the big 'hds' group is ~53%
    tags = ["swgw", "prowl", "flowout", "dtred", "fdc", "longtt", "ltdiff", "simhk", "consec", "thresh", "malf"]
    gnames = ["hds", "vol", "flux"] + ["oname:{0}_otype:lst_usecol:grp{1:04d}".format(tags[i % len(tags)], i)
                                       for i in range(ngroups - 3)]
    n_hds = int(0.533 * nobs)
    n_vf = int(0.0153 * nobs)
    grp = np.empty(nobs, dtype=object)
    grp[:n_hds] = "hds"
    grp[n_hds:n_hds + n_vf] = "vol"
    grp[n_hds + n_vf:n_hds + 2 * n_vf] = "flux"
    rest = nobs - (n_hds + 2 * n_vf)
    grp[n_hds + 2 * n_vf:] = np.array(gnames[3:], dtype=object)[np.arange(rest) % (ngroups - 3)]
    onames = ["oname:{0}_otype:lst_usecol:trgw-0-{1:07d}_time:{2:.1f}_extra:padpadpad".format(
        "hds" if g == "hds" else "obs", i, 10.0 * (i % 500)) for i, g in enumerate(grp)]
    weight = np.zeros(nobs)
    # weighted obs spread across the tagged groups so the phi factor file has something to work on
    nz_idx = rng.choice(np.arange(n_hds + 2 * n_vf, nobs), size=min(n_nz, rest), replace=False)
    weight[nz_idx] = 1.0
    cols = ["obsnme", "obsval", "weight", "obgnme", "datetime", "days", "easting", "file", "i", "idx", "j", "k",
            "kper", "lrc", "northing", "obshk", "obsss", "oname", "otype", "reachid", "site", "usecol",
            "wateryear", "wellk", "zone", "observed", "obs_count", "sfrsite", "standard_deviation",
            "lower_bound", "sval", "eval", "start_datetime", "end_datetime"]
    odf = pd.DataFrame({c: "" for c in cols}, index=np.arange(nobs))
    odf["obsnme"] = onames
    odf["obsval"] = rng.normal(10.0, 1.0, nobs)
    odf["weight"] = weight
    odf["obgnme"] = grp
    odf["i"] = np.arange(nobs) % 120
    odf["j"] = np.arange(nobs) % 90
    odf["k"] = np.arange(nobs) % 3
    odf["oname"] = np.where(grp == "hds", "hds", "obs")
    odf["otype"] = "lst"
    odf["usecol"] = grp
    odf["observed"] = "False"
    odf["obs_count"] = 0.0
    std_idx = rng.choice(nz_idx, size=min(n_std, len(nz_idx)), replace=False)
    sd = np.full(nobs, "", dtype=object)
    sd[std_idx] = 0.5
    odf["standard_deviation"] = sd
    lb = np.full(nobs, "", dtype=object)
    lb[rng.choice(np.arange(nobs), size=n_lb, replace=False)] = -1.0e10
    odf["lower_bound"] = lb
    odf.to_csv("mimic.obs_data.csv", index=False)

    # model outputs: split the obs across n_out_files files, one value per line
    splits = np.array_split(np.arange(nobs), n_out_files)
    ins_rows = []
    for ifile, idx in enumerate(splits):
        with open("out{0:02d}.csv.ins".format(ifile), "w") as f:
            f.write("pif ~\n")
            for i in idx:
                f.write("l1 !{0}!\n".format(onames[i]))
        ins_rows.append(("out{0:02d}.csv.ins".format(ifile), "out{0:02d}.csv".format(ifile)))
    pd.DataFrame(ins_rows, columns=["pest_file", "model_file"]).to_csv("mimic.insfile_data.csv", index=False)
    sizes = [len(s) for s in splits]
    with open("forward_run.py", "w") as f:
        f.write("import numpy as np\n")
        f.write("sizes = {0}\n".format(sizes))
        f.write("for i, n in enumerate(sizes):\n")
        f.write("    np.savetxt('out{0:02d}.csv'.format(i), 10.0 + 0.01 * np.sin(np.arange(n) + i), fmt='%15.6e')\n")

    # prior parameter ensemble as a binary file, like prior_pe.bin
    pe = pd.DataFrame(np.exp(rng.normal(0.0, 0.3, (n_prior_reals, npar))), columns=pnames)
    # to_coo is the long-name binary format, the same one the real prior_pe.bin is in
    pyemu.Matrix(x=pe.values, row_names=[str(i) for i in range(n_prior_reals)], col_names=pnames).to_coo(
        "prior_pe.bin")

    with open("phi.csv", "w") as f:
        for t, v in zip(tags, [0.1, 0.05, 0.15, 0.05, 0.2, 1e-10, 0.05, 1e-6, 0.2, 0.1, 0.2]):
            f.write("{0},{1}\n".format(t, v))

    with open("mimic.pst", "w") as f:
        f.write("pcf version=2\n* control data keyword\n")
        opts = [("pestmode", "estimation"), ("noptmax", -2), ("svdmode", 1), ("maxsing", 10000000),
                ("eigthresh", 1e-6), ("eigwrite", 1), ("ies_par_en", "prior_pe.bin"),
                ("ies_drop_conflicts", "False"), ("ies_bad_phi_sigma", 1.3), ("ies_num_reals", num_reals),
                ("save_binary", "true"), ("save_dense", "true"), ("overdue_giveup_fac", 100),
                ("ies_phi_factor_file", "phi.csv"), ("ies_bad_phi", 1e40), ("ies_multimodal_alpha", 0.25),
                ("ies_subset_size", -10), ("ies_initial_lambda", -15), ("ies_n_iter_mean", 999),
                ("panther_agent_freeze_on_fail", "False"), ("ies_lambda_mults", "0.1,1.0,10.0"),
                ("lambda_scale_fac", "0.1,0.25,0.75,1.0"), ("ies_num_threads", 10)]
        for k, v in opts:
            f.write("{0:<30s} {1}\n".format(k, v))
        f.write("* parameter groups external\nmimic.pargp_data.csv\n")
        f.write("* parameter data external\nmimic.par_data.csv\n")
        f.write("* observation data external\nmimic.obs_data.csv\n")
        f.write("* model command line\npython forward_run.py\n")
        f.write("* model input external\nmimic.tplfile_data.csv\n")
        f.write("* model output external\nmimic.insfile_data.csv\n")
    print("nobs", nobs, "npar", npar, "ngroups", ngroups, "nnz", int(weight.sum()))


if __name__ == "__main__":
    main(scale=float(sys.argv[1]) if len(sys.argv) > 1 else 0.1)
