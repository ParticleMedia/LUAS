#!/bin/bash
set -x

#for num in {1..50}
#do
#  echo $num
#  python run_single.py
#  sleep 30s
#done

mkdir -p logs

for n in {1..1}
do
  ts=$(date +"%Y-%m-%d_%H-%M")
  for part in {0..63}
  do
  python run.py --part ${part} > "logs/single.${ts}.${part}.txt" 2>&1 &
  done
  wait
done