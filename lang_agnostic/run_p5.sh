P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev(){ local m=$1 s=$2 d=runs/${1//:/_}/$2
  $P harness3.py $s $m > logs/p5gen_${m//:/_}_$s.log 2>&1 || { echo "FAIL gen $m $s"; tail -5 logs/p5gen_${m//:/_}_$s.log; return; }
  grep -E "gate fire" logs/p5gen_${m//:/_}_$s.log
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/p5judge_${m//:/_}_$s.log 2>&1 || echo "FAIL judge $m $s"
  $P metrics.py $m $s --vs base | grep -E "^==|vs|harmless"; }
date; $P p5_camo.py; $P p5_cal.py 2>&1 | grep -vE "Loading|Warning|warn" | tail -8 || { echo FAIL cal; exit 1; }
for m in base xgate:0.15 spangate:0.15 delgate:0.15 delspan:0.15; do echo "=== $m camo $(date +%H:%M)"; ev $m camo; done
echo CAMO DONE
for m in spangate:0.15 delgate:0.15 delspan:0.15; do echo "=== $m sub_test $(date +%H:%M)"; ev $m sub_test; done
echo P5 DONE; date
