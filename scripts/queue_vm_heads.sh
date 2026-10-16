#!/bin/bash
# GPU runs of the matched read-out heads and the attention cut inside Depth Anything V2, on the course A10 (same
# conventions as queue_vm.sh). Every step appends to queue.log.
set -u
PY=/home/student/anaconda3/envs/stud_prj/bin/python
[ -x "$PY" ] || { echo "queue_vm_heads.sh: $PY not found; this script only runs on the course VM (see README)" >&2; exit 1; }
cd ~/penrose || { echo "queue_vm_heads.sh: no ~/penrose; this script only runs on the course VM (see README)" >&2; exit 1; }
export PYTHONWARNINGS=ignore
log() { echo "[$(date +%H:%M:%S)] $*" >> queue.log; }
FAILED=0
run() { "$PY" "$@" >> queue.log 2>&1 || { log "FAILED (exit $?): $*"; echo "queue_vm_heads.sh: failed: $*" >&2; FAILED=$((FAILED + 1)); }; }

# 1. the training scenes (possible objects only; pdepth/scenes.py)
log "start scenes"; run scripts/make_scenes.py data/scenes 1500 32; log "end scenes"
# 2. matched heads on the frozen DA-V2-S backbone, same scenes, two seeds: depth (0-form) vs depth gradient (1-form)
log "start heads bench"; run scripts/train_heads.py 0form bench --max-steps 60 --log-every 20; log "end heads bench"
for seed in 0 1; do
  for kind in 0form 1form; do
    log "start head $kind s$seed"; run scripts/train_heads.py $kind ${kind}-s-seed$seed --seed $seed; log "end head $kind s$seed"
    log "start run head $kind s$seed"; run scripts/run_heads.py ${kind}-s-seed$seed; log "end run head $kind s$seed"
  done
done
# 3. attention cut inside the network (pixels untouched), impossible images and controls
for m in dav2-s dav2-b dav2-l; do
  log "start attn $m"; run scripts/run_attention_cut.py $m "data/stim2/[AB]_*.npz" keys; log "end attn $m"
done
log "queue_vm_heads.sh done, $FAILED failed step(s)"
