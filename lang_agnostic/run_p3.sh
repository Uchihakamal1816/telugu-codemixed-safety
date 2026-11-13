#!/bin/bash
# P3 overnight: literature methods on seed-0 subsets of TEST (sub_test) and UNSEEN (sub_unseen). Resumable; failures logged.
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev(){ local m=$1 s=$2 d=runs/${1//:/_}/$2
  $P harness3.py $s $m > logs/p3gen_${m//:/_}_$s.log 2>&1 || { echo "FAIL gen $m $s"; tail -5 logs/p3gen_${m//:/_}_$s.log; return; }
  grep -E "gate fire|routes" logs/p3gen_${m//:/_}_$s.log
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/p3judge_${m//:/_}_$s.log 2>&1 || echo "FAIL judge $m $s"
  $P metrics.py $m $s --vs base; }
cap(){ [ -f runs/${1//:/_}/capability.json ] || $P capability3.py "$1" > logs/p3cap_${1//:/_}.log 2>&1 || echo "FAIL cap $1"; cat runs/${1//:/_}/capability.json; echo; }
date; $P p3_prep.py 2>&1 | grep -v Warn
$P metrics.py base sub_test; $P metrics.py base sub_unseen; $P metrics.py steer:d_en:0.05 sub_test --vs base; $P metrics.py steer:d_en:0.05 sub_unseen --vs base
[ -d runs/p3_tli_lora ] || $P p3_tli.py > logs/p3_tli.log 2>&1 || echo FAIL tli; grep -v Load logs/p3_tli.log | tail -8
[ -d runs/p3_path_lora ] || $P p3_path.py > logs/p3_path.log 2>&1 || echo FAIL path; grep -v Load logs/p3_path.log | tail -8
for m in twostage:11:0.05 twostage:8:0.05 twostage:11:0.1 formsteer:0.05 tli tli_steer:0.05 path pivot pivot_steer:0.05; do
  echo "=== $m $(date +%H:%M)"; for s in sub_test sub_unseen; do ev $m $s; done; done
for m in twostage:11:0.05 twostage:11:0.1 formsteer:0.05 tli tli_steer:0.05 path; do cap $m; done
echo P3 DONE; date
