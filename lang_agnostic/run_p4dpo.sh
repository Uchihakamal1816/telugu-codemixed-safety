P=~/lexsi-hackathon/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date; rm -rf runs/p4_dpo_lora runs/dpo
$P p4_train.py dpo > logs/p4_dpo.log 2>&1 || { echo FAIL dpo; tail -3 logs/p4_dpo.log; exit 1; }
grep -E "pairs|epoch" logs/p4_dpo.log
ev(){ local m=$1 s=$2 d=runs/${1//:/_}/$2
  $P harness3.py $s $m > logs/p4gen_${m//:/_}_$s.log 2>&1 || { echo "FAIL gen $m $s"; tail -5 logs/p4gen_${m//:/_}_$s.log; return; }
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/p4judge_${m//:/_}_$s.log 2>&1 || echo "FAIL judge $m $s"
  $P metrics.py $m $s --vs base | grep -E "^==|vs|harmless"; }
ev dpo sub_test
$P capability3.py dpo > logs/p4cap_dpo.log 2>&1 || echo "FAIL cap"; echo "dpo $(cat runs/dpo/capability.json)"
ev dpo sub_unseen; echo DPO DONE; date
