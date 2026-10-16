#!/bin/bash
# GPU read-outs of the ten depth networks on the course A10, in priority order. Every step appends to queue.log.
# The interpreter and folder below are the course VM's; on another machine run the per-model commands directly
# (see "Reproduce" in README.md).
# S1: the samplers' seed subset (A: 8 layouts x chirality, all of B, C, D).
set -u
PY=/home/student/anaconda3/envs/stud_prj/bin/python
[ -x "$PY" ] || { echo "queue_vm.sh: $PY not found; this script only runs on the course VM (see README)" >&2; exit 1; }
cd ~/penrose || { echo "queue_vm.sh: no ~/penrose; this script only runs on the course VM (see README)" >&2; exit 1; }
export PYTHONWARNINGS=ignore
log() { echo "[$(date +%H:%M:%S)] $*" >> queue.log; }
# a failed step is logged and counted, and the queue goes on to the next one
FAILED=0
run() { "$PY" "$@" >> queue.log 2>&1 || { log "FAILED (exit $?): $*"; echo "queue_vm.sh: failed: $*" >&2; FAILED=$((FAILED + 1)); }; }

mkdir -p sel_S1
rm -f sel_S1/*
for a in 000 015 030 045 060 075 090 105; do
  ln -sf ../data/stim2/A_a${a}_p0_t0.npz sel_S1/; ln -sf ../data/stim2/A_a${a}_p0_t0_m.npz sel_S1/
done
for f in data/stim2/B_*.npz data/stim2/C_*.npz data/stim2/D_*.npz; do ln -sf ../$f sel_S1/; done
log "S1 $(ls sel_S1 | wc -l) files"

# 1. deterministic / one-step models on all 460 stimuli
for m in dav2-s dav2-b dav2-l dpt-l zoe lotus-d e2eft; do
  log "start $m"; run scripts/run_models.py $m "data/stim2/*.npz"; log "end $m"
done
# 2. one-step seed runs on S1 (32 seeds): Lotus-G, Marigold v1-1 (zero terminal SNR) and v1-0
log "start lotus-g seeds"; run scripts/run_models.py lotus-g "sel_S1/*.npz" 32; log "end lotus-g"
log "start mg11 1-step"; run scripts/run_marigold.py mg11_1step "sel_S1/*.npz" 1 32 --model prs-eth/marigold-depth-v1-1; log "end mg11 1-step"
log "start mg10 1-step"; run scripts/run_marigold.py mg10_1step "sel_S1/*.npz" 1 32 --model prs-eth/marigold-depth-v1-0; log "end mg10 1-step"
log "ALL DONE"
[ "$FAILED" -eq 0 ] || { echo "queue_vm.sh: $FAILED step(s) failed, see queue.log" >&2; exit 1; }
