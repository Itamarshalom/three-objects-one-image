#!/bin/bash
# GPU runs of the erasure probe and the normal networks on the course A10 (same conventions as queue_vm.sh).
# Every step appends to queue.log. P: the 16 impossible images of the samplers' subset (8 layouts x chirality) and
# all 48 possible controls. Rings: the closed possible loops of data/loops (scripts/make_loops.py).
set -u
PY=/home/student/anaconda3/envs/stud_prj/bin/python
[ -x "$PY" ] || { echo "queue_vm_probe.sh: $PY not found; this script only runs on the course VM (see README)" >&2; exit 1; }
cd ~/penrose || { echo "queue_vm_probe.sh: no ~/penrose; this script only runs on the course VM (see README)" >&2; exit 1; }
export PYTHONWARNINGS=ignore
log() { echo "[$(date +%H:%M:%S)] $*" >> queue.log; }
FAILED=0
run() { "$PY" "$@" >> queue.log 2>&1 || { log "FAILED (exit $?): $*"; echo "queue_vm_probe.sh: failed: $*" >&2; FAILED=$((FAILED + 1)); }; }

mkdir -p sel_P
rm -f sel_P/*
for a in 000 015 030 045 060 075 090 105; do
  ln -sf ../data/stim2/A_a${a}_p0_t0.npz sel_P/; ln -sf ../data/stim2/A_a${a}_p0_t0_m.npz sel_P/
done
for f in data/stim2/B_*.npz; do ln -sf ../$f sel_P/; done
log "P $(ls sel_P | wc -l) files, rings $(ls data/loops | wc -l)"

# 1. normal networks that share a depth network's backbone (all impossible images and controls)
for m in e2eft-n lotus-d-n; do log "start $m"; run scripts/run_normals.py $m "data/stim2/[AB]_*.npz"; log "end $m"; done
log "start mg11-n"; run scripts/run_normals.py mg11-n "data/stim2/[AB]_*.npz" 4; log "end mg11-n"
# 2. erasure probe, deterministic depth networks, all impossible images and controls
for m in lotus-d e2eft dpt-l zoe dav2-l; do log "start probe $m"; run scripts/run_probe.py $m "data/stim2/[AB]_*.npz"; log "end probe $m"; done
# 3. closed possible rings, deterministic networks
for m in dav2-b dav2-l lotus-d e2eft dpt-l zoe; do
  log "start rings $m"; PROBE_TAG=_loops run scripts/run_probe.py $m "data/loops/*.npz"; log "end rings $m"
done
# 4. erasure probe, samplers (one step, 8 noise seeds) on P; rings with 4 seeds
for m in mg11-1 lotus-g mg10-1; do log "start probe $m"; run scripts/run_probe.py $m "sel_P/*.npz" 8; log "end probe $m"; done
for m in mg11-1 lotus-g mg10-1; do
  log "start rings $m"; PROBE_TAG=_loops run scripts/run_probe.py $m "data/loops/*.npz" 4; log "end rings $m"
done
log "queue_vm_probe.sh done, $FAILED failed step(s)"
