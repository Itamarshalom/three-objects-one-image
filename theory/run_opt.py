"""Run every population optimisation of check3b_opt.py (Depth Anything V2, MiDaS and E2E-FT losses) and
check7_dav2trim.py (Depth Anything V2 with 10% trimming, two variants) at the four beliefs p used in docs/theory.md,
JOBS at a time with NT torch threads each, then write opt/summary.txt.
Logs: opt/log_<loss>_<p>.txt and opt7/log_<loss>_<p>.txt.  8-20 min with JOBS=6, NT=3 on a 22-thread laptop CPU,
depending on what else is running.
usage: python run_opt.py [JOBS] [NT]"""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NT = sys.argv[2] if len(sys.argv) > 2 else "3"
PS = ["0.6,0.3,0.1", "0.4,0.35,0.25", "0.3333333,0.3333333,0.3333334", "0.45,0.45,0.1"]
for d in ("opt", "opt7"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)


def run(args, log):
    with open(os.path.join(HERE, log), "w") as fh:
        r = subprocess.run([sys.executable] + args, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                           env=dict(os.environ, NT=NT))
    print(("ok    " if r.returncode == 0 else "FAILED") + " " + " ".join(args), flush=True)
    return r.returncode


# the dav2 and midas losses are the slow ones, so they start first
jobs = [(["check3b_opt.py", L, p, "400"], os.path.join("opt", f"log_{L}_{p}.txt")) for L in ("dav2", "midas", "lsl1") for p in PS]
with ThreadPoolExecutor(JOBS) as ex:
    codes = list(ex.map(lambda j: run(*j), jobs))
jobs = [(["check7_dav2trim.py", p, "400", L], os.path.join("opt7", f"log_{L}_{p}.txt")) for L in ("dav2trim", "dav2trimgm") for p in PS]
with ThreadPoolExecutor(JOBS) as ex:
    codes += list(ex.map(lambda j: run(*j), jobs))
with open(os.path.join(HERE, "opt", "summary.txt"), "w") as fh:
    subprocess.run([sys.executable, "summarize_opt.py"], cwd=HERE, stdout=fh, check=True)
sys.exit(max(codes))
