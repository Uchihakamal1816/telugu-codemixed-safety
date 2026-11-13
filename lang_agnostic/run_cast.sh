#!/bin/bash
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f runs/cast_cond.pt ] || $P p2_cast_fit.py > logs/cast_fit.log 2>&1; echo "fit exit $?"; grep -v Loading logs/cast_fit.log | tail -25
for a in 0.1 0.2 0.3; do m=cast:d_multi:$a; d=runs/${m//:/_}/dev
  $P harness.py dev $m > logs/gen_${m//:/_}.log 2>&1 || { echo FAIL gen $m; tail -5 logs/gen_${m//:/_}.log; continue; }
  $P judge.py $d/generations.jsonl $d/judged.jsonl > /dev/null 2>&1; $P metrics.py $m dev --vs base | head -1; $P metrics.py $m dev --vs base | tail -1
  $P capability.py $m 2>/dev/null | tail -1; done; echo CAST DONE
