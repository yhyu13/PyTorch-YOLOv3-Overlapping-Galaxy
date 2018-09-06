#!/bin/bash

train(){
    python train.py
}

until train; do
    echo "'train' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
