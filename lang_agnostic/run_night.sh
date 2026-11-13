#!/bin/bash
# Overnight: base dev, diagnosis, steering sweep, CAST. Each step resumable; failures logged, not fatal.
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev() { local m=$1; local d=runs/${m//:/_}/dev
  $P harness.py dev "$m" > logs/gen_${m//:/_}.log 2>&1 || echo "FAIL gen $m"
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/judge_${m//:/_}.log 2>&1 || echo "FAIL judge $m"
  [ "$m" = base ] && $P metrics.py base dev || $P metrics.py "$m" dev --vs base
  [ -f runs/${m//:/_}/capability.json ] || $P capability.py "$m" > logs/cap_${m//:/_}.log 2>&1 || echo "FAIL cap $m"
  cat runs/${m//:/_}/capability.json; echo; date; }
date; ev base
[ -f acts/train.pt ] || $P p1_acts.py train > logs/acts_train.log 2>&1
[ -f acts/dev.pt ] || $P p1_acts.py dev > logs/acts_dev.log 2>&1
$P p1_probes.py > logs/probes.log 2>&1; cat logs/probes.log | tail -6
[ -f runs/dirs.pt ] || $P p2_dirs.py
for d in d_en d_multi d_multi_nolang; do for a in 0.05 0.1 0.15; do ev steer:$d:$a; done; done
[ -f runs/cast_cond.pt ] || $P p2_cast_fit.py > logs/cast_fit.log 2>&1; tail -20 logs/cast_fit.log
for a in 0.1 0.2 0.3; do ev cast:d_multi:$a; done
echo NIGHT DONE; date
