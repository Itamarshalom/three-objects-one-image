#!/bin/bash
# Run after queue_vm.sh: possible controls under the multi-step sampler settings (references for the excess tear),
# plus the v1.0 gap morph. Waits until queue_vm.sh has logged ALL DONE.
# The interpreter and folder below are the course VM's; on another machine run the per-model commands directly
# (see "Model runs" in README.md).
set -u
PY=/home/student/anaconda3/envs/stud_prj/bin/python
[ -x "$PY" ] || { echo "queue_vm3.sh: $PY not found; this script only runs on the course VM (see README)" >&2; exit 1; }
cd ~/penrose || { echo "queue_vm3.sh: no ~/penrose; this script only runs on the course VM (see README)" >&2; exit 1; }
export PYTHONWARNINGS=ignore
log() { echo "[$(date +%H:%M:%S)] $*" >> queue.log; }
# a failed step is logged and counted, and the queue goes on to the next one
FAILED=0
run() { "$PY" "$@" >> queue.log 2>&1 || { log "FAILED (exit $?): $*"; echo "queue_vm3.sh: failed: $*" >&2; FAILED=$((FAILED + 1)); }; }
until grep -q "ALL DONE" queue.log; do sleep 60; done
mkdir -p sel_R sel_M
rm -f sel_R/* sel_M/*
for a in 000 030 060 090; do
  for f in data/stim2/B_a${a}_k*.npz; do ln -sf ../$f sel_R/; done
done
for f in data/stim2/C_a000_*.npz data/stim2/C_a060_*.npz; do ln -sf ../$f sel_M/; done
log "start refs mg11 multi"; run scripts/run_marigold.py mg11_multi_refs "sel_R/*.npz" 4,10 16 --model prs-eth/marigold-depth-v1-1
log "start refs mg10 multi"; run scripts/run_marigold.py mg10_multi_refs "sel_R/*.npz" 10 16 --model prs-eth/marigold-depth-v1-0
log "start mg10 morph multi"; run scripts/run_marigold.py mg10_multi_morph "sel_M/*.npz" 10 16 --model prs-eth/marigold-depth-v1-0
log "QUEUE3 DONE"
[ "$FAILED" -eq 0 ] || { echo "queue_vm3.sh: $FAILED step(s) failed, see queue.log" >&2; exit 1; }
