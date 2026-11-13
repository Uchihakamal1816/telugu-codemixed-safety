#!/bin/bash
cd ~/lexsi-hackathon/lang_agnostic; P=~/lexsi-hackathon/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$($P -c "from huggingface_hub import snapshot_download as s;print(s(\"manas2002/aya-indicguard-hardened\",revision=\"63fff8eb7a6c96cbc2ddd9c86c1be2c04a2b6f80\"))") || { echo DL FAIL; exit 1; }
export MODEL_PATH=$D; m=ft:hardened
for s in test unseen; do d=runs/ft_hardened/$s
  $P harness.py $s $m > logs/gen_ft_hardened_$s.log 2>&1 || { echo FAIL gen $s; tail -3 logs/gen_ft_hardened_$s.log; continue; }
  $P judge.py $d/generations.jsonl $d/judged.jsonl > logs/judge_ft_hardened_$s.log 2>&1
  $P metrics.py $m $s --vs base; done
$P capability.py $m | tail -1
echo HARDENED DONE
