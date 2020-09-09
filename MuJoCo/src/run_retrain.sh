#!/usr/bin/env bash
for i in 0 1 2 3 4; do
    python victim_train.py 2 > console_$i.txt &
    sleep 10
done
