#! /bin/bash

squeue -u $USER | grep 197 | awk '{print $1}' | xargs -n 1 scancel