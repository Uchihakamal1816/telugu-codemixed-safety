#!/bin/bash
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev(){ local m=$1 s=$2 d=runs/${1//:/_}/$2
  $P harness3.py $s $m > logs/p4gen_${m//:/_}_$s.log 2>&1 || { echo "FAIL gen $m $s"; tail -5 logs/p4gen_${m//:/_}_$s.log; return; }
  grep -E "gate fire" logs/p4gen_${m//:/_}_$s.log
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/p4judge_${m//:/_}_$s.log 2>&1 || echo "FAIL judge $m $s"
  $P metrics.py $m $s --vs base | grep -E "^==|vs|harmless"; }
cap(){ $P capability3.py "$1" > logs/p4cap_${1//:/_}.log 2>&1 || echo "FAIL cap $1"; echo "$1 $(cat runs/${1//:/_}/capability.json)"; }
date
[ -f runs/p4_swe.pt ] || $P p4_swe.py > logs/p4_swe.log 2>&1 || echo FAIL swe; grep -E "^L|keys" logs/p4_swe.log
echo "=== swe:0.1 $(date +%H:%M)"; ev swe:0.1 sub_test
echo "=== xgate:0.15 $(date +%H:%M)"; ev xgate:0.15 sub_test
[ -f runs/p4_prefs.json ] || $P p4_prefs.py > logs/p4_prefs.log 2>&1 || echo FAIL prefs
[ -d runs/p4_dpo_lora ] || $P p4_train.py dpo > logs/p4_dpo.log 2>&1 || echo FAIL dpo; grep -E "pairs|epoch" logs/p4_dpo.log
echo "=== dpo $(date +%H:%M)"; ev dpo sub_test
[ -d runs/p4_nlora_lora ] || $P p4_train.py nlora > logs/p4_nlora.log 2>&1 || echo FAIL nlora; grep -E "sft|epoch" logs/p4_nlora.log
echo "=== nlora $(date +%H:%M)"; ev nlora sub_test
echo "=== translit:0.05 $(date +%H:%M)"; ev translit:0.05 sub_test
echo SUBTEST DONE $(date +%H:%M)
for m in swe:0.1 dpo nlora; do cap $m; done
for m in swe:0.1 xgate:0.15 dpo nlora translit:0.05; do echo "=== $m unseen"; ev $m sub_unseen; done
echo P4 DONE; date
