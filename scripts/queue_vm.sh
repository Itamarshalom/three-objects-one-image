#!/bin/bash
# Full GPU run on the course A10, in priority order. Every step appends to queue.log.
# The interpreter and folder below are the course VM's; on another machine run the per-model commands directly
# (see "Model runs" in README.md).
# S1: seeds subset (A: 8 layouts x chirality, all of B, C, D); S2: multi-step subset.
set -u
PY=/home/student/anaconda3/envs/stud_prj/bin/python
[ -x "$PY" ] || { echo "queue_vm.sh: $PY not found; this script only runs on the course VM (see README)" >&2; exit 1; }
cd ~/penrose || { echo "queue_vm.sh: no ~/penrose; this script only runs on the course VM (see README)" >&2; exit 1; }
export PYTHONWARNINGS=ignore
log() { echo "[$(date +%H:%M:%S)] $*" >> queue.log; }
# a failed step is logged and counted, and the queue goes on to the next one
FAILED=0
run() { "$PY" "$@" >> queue.log 2>&1 || { log "FAILED (exit $?): $*"; echo "queue_vm.sh: failed: $*" >&2; FAILED=$((FAILED + 1)); }; }

mkdir -p sel_S1 sel_S2 sel_T
rm -f sel_S1/* sel_S2/* sel_T/*
for a in 000 015 030 045 060 075 090 105; do
  ln -sf ../data/stim2/A_a${a}_p0_t0.npz sel_S1/; ln -sf ../data/stim2/A_a${a}_p0_t0_m.npz sel_S1/
  ln -sf ../data/stim2/A_a${a}_p0_t0.npz sel_S2/
done
for f in data/stim2/B_*.npz data/stim2/C_*.npz data/stim2/D_*.npz; do ln -sf ../$f sel_S1/; done
for f in data/stim2/C_a000_*.npz data/stim2/C_a060_*.npz; do ln -sf ../$f sel_S2/; done
for a in 000 030 060 090; do ln -sf ../data/stim2/A_a${a}_p0_t0.npz sel_T/; done
log "S1 $(ls sel_S1 | wc -l) files, S2 $(ls sel_S2 | wc -l), T $(ls sel_T | wc -l)"

# 1. deterministic / one-step models on all 460 stimuli
for m in dav2-s dav2-b dav2-l dpt-l zoe lotus-d e2eft; do
  log "start $m"; run scripts/run_models.py $m "data/stim2/*.npz"; log "end $m"
done
# 2. one-step seed runs on S1 (32 seeds): Lotus-G, Marigold v1-1 (zero terminal SNR) and v1-0
log "start lotus-g seeds"; run scripts/run_models.py lotus-g "sel_S1/*.npz" 32; log "end lotus-g"
log "start mg11 1-step"; run scripts/run_marigold.py mg11_1step "sel_S1/*.npz" 1 32 --model prs-eth/marigold-depth-v1-1; log "end mg11 1-step"
log "start mg10 1-step"; run scripts/run_marigold.py mg10_1step "sel_S1/*.npz" 1 32 --model prs-eth/marigold-depth-v1-0; log "end mg10 1-step"
# 3. multi-step sampling (64 seeds x {4,10} steps) on S2, v1-1; v1-0 contrast on the A part
log "start mg11 multi"; run scripts/run_marigold.py mg11_multi "sel_S2/*.npz" 4,10 64 --model prs-eth/marigold-depth-v1-1; log "end mg11 multi"
log "start mg10 multi"; run scripts/run_marigold.py mg10_multi "sel_S2/A_*.npz" 10 64 --model prs-eth/marigold-depth-v1-0; log "end mg10 multi"
# 4. x0-prediction trajectories (does a sample start paying a tear along the way?), v1-1 and v1-0, 10 steps
log "start traj"; run scripts/run_marigold.py mg11_traj "sel_T/*.npz" 10 32 --traj --model prs-eth/marigold-depth-v1-1
run scripts/run_marigold.py mg10_traj "sel_T/*.npz" 10 32 --traj --model prs-eth/marigold-depth-v1-0; log "end traj"
log "ALL DONE"
[ "$FAILED" -eq 0 ] || { echo "queue_vm.sh: $FAILED step(s) failed, see queue.log" >&2; exit 1; }
