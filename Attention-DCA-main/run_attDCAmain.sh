#!/usr/bin/env bash

for d in 5 10 23 50 100 150 200
do
  python3 ./CODE/AttentionDCA_python/src/attDCA_main.py 180 "$d"
done
