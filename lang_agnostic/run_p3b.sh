#!/bin/bash
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev(){ local m=$1 s=$2 d=runs/${1//:/_}/$2
  $P harness3.py $s $m > logs/p3gen_${m//:/_}_$s.log 2>&1 || { echo "FAIL gen $m $s"; tail -5 logs/p3gen_${m//:/_}_$s.log; return; }
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/p3judge_${m//:/_}_$s.log 2>&1 || echo "FAIL judge $m $s"
  $P metrics.py $m $s --vs base; }
date; [ -d runs/p3_path2_lora ] || $P p3_path2.py > logs/p3_path2.log 2>&1 || echo FAIL path2; grep -E "steered|train ex|epoch|trainable" logs/p3_path2.log
for m in path2 pivotback_steer:0.05; do echo "=== $m $(date +%H:%M)"; for s in sub_test sub_unseen; do ev $m $s; done; done
$P capability3.py path2 > logs/p3cap_path2.log 2>&1; cat runs/path2/capability.json; echo; echo P3B DONE; date
