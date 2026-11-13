#!/bin/bash
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev(){ local m=$1 s=$2 d=runs/${1//:/_}/$2
  $P harness.py $s $m > logs/gen_${m//:/_}_$s.log 2>&1 || { echo FAIL gen $m $s; tail -3 logs/gen_${m//:/_}_$s.log; return; }
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/judge_${m//:/_}_$s.log 2>&1
  if [ $m = base ]; then $P metrics.py $m $s | head -1; else $P metrics.py $m $s --vs base | head -1; $P metrics.py $m $s --vs base | tail -1; fi; }
[ -f runs/cast_cond2.pt ] || $P p2_gate2.py > logs/gate2.log 2>&1; echo "gate2 exit $?"; grep -v Loading logs/gate2.log | tail -20
for m in cast2:d_multi:0.2 cast2:d_multi:0.3; do ev $m dev; $P capability.py $m 2>/dev/null | tail -1; done
echo DEV DONE
for s in test unseen; do for m in base steer:d_en:0.05 steer:d_multi:0.05 cast:d_multi:0.1 cast2:d_multi:0.2 cast2:d_multi:0.3; do ev $m $s; done; done
echo FINAL DONE
