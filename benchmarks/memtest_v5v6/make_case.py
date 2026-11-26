"""build a big-obs test case for comparing master memory between pestpp-ies versions.

nobs observations, 10 parameters, a model that just writes nobs numbers. writes two
control files with pyemu that differ only in how the sections are stored:
  inline.pst   - version 1, everything in the control file
  external.pst - version 2, observation/parameter data in external csv files
"""
import os
import sys
import numpy as np
import pyemu


def main(nobs=500000, ngroups=20, nz_frac=0.1, npar=10, num_reals=50):
    d = os.path.dirname(os.path.abspath(__file__))
    os.chdir(d)
    rng = np.random.default_rng(11)

    pnames = ["p{0:02d}".format(i) for i in range(npar)]
    # names about as long as pstfrom names
    onames = ["oname:hds_otype:lst_usecol:trgw-{0:07d}_time:{1:.1f}".format(i, 10.0 * (i % 50))
              for i in range(nobs)]

    with open("model.in.tpl", "w") as f:
        f.write("ptf ~\n")
        for p in pnames:
            f.write("{0} ~   {0}   ~\n".format(p))
    with open("model.in", "w") as f:
        for p in pnames:
            f.write("{0} 1.0\n".format(p))
    with open("model.out.ins", "w") as f:
        f.write("pif ~\n")
        for o in onames:
            f.write("l1 !{0}!\n".format(o))
    with open("forward_run.py", "w") as f:
        f.write("import numpy as np\n")
        f.write("p = np.loadtxt('model.in', usecols=1)\n")
        f.write("n = {0}\n".format(nobs))
        f.write("v = 10.0 + p.sum() * 0.01 * np.sin(np.arange(n))\n")
        f.write("np.savetxt('model.out', v, fmt='%15.6e')\n")
    os.system("python forward_run.py")

    pst = pyemu.Pst.from_io_files("model.in.tpl", "model.in", "model.out.ins", "model.out", pst_path=".")
    par = pst.parameter_data
    par.loc[:, "parval1"] = 1.0
    par.loc[:, "parlbnd"] = 0.1
    par.loc[:, "parubnd"] = 10.0
    par.loc[:, "partrans"] = "none"
    obs = pst.observation_data
    obs.loc[:, "obsval"] = rng.normal(10.0, 1.0, nobs)
    obs.loc[:, "weight"] = np.where(rng.random(nobs) < nz_frac, 1.0, 0.0)
    obs.loc[:, "obgnme"] = ["grp{0:02d}".format(i % ngroups) for i in range(nobs)]
    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = -2
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.pestpp_options["overdue_giveup_fac"] = 1000
    pst.write("inline.pst", version=1)
    pst.write("external.pst", version=2)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    main(nobs=n)
