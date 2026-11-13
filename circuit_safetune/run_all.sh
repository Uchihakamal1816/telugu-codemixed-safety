#!/bin/bash
set -e
cd ~/lexsi-hackathon/circuit_safetune; PY=~/lexsi-hackathon/.venv/bin/python
[ -f direction.pt ] || $PY a_extract.py
[ -f circuit.pt ] || $PY b_circuit.py
[ -f generations.jsonl ] || $PY c_steer.py 0 ${STRENGTHS:-0.5 1.0}
$PY d_judge.py generations.jsonl judged.jsonl
echo ALL DONE
