#!/usr/bin/env bash

for i in 0 1 2 3 4 5; do
    python play_pong_retrain.py --hyper_index $i > console_$i.txt &
    sleep 10
done
