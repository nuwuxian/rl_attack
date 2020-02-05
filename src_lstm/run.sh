#!/usr/bin/env bash
for i in 0; do
    python adv_train.py 5 > console_$i.txt &
    sleep 10
done
