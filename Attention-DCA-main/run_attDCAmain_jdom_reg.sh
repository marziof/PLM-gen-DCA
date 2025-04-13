#!/usr/bin/env bash

for l in 0.005 0.01 0.05 0.1
do
  python3 ./CODE/AttentionDCA_python/src/attDCA_main.py 50 23 "$l"
done
