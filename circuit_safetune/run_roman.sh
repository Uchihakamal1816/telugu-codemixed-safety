#!/bin/bash
# Romanized-Telugu direction: extract at the English run's layer, steer, judge. Strength 0 is shared with the English run.
set -e
cd ~/lexsi-hackathon/circuit_safetune; PY=~/lexsi-hackathon/.venv/bin/python; export TAG=_roman PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f direction_roman.pt ] || PICK_LAYER=18 $PY a_extract.py
[ -f generations_roman.jsonl ] || $PY c_steer.py 4 10 40
$PY d_judge.py generations_roman.jsonl judged_roman.jsonl
echo ROMAN DONE
