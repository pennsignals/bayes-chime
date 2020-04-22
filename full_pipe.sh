#!/bin/bash
python _01_GOF_sims.py -p data/$1_parameters.csv -t data/$1_ts.csv -o $1 | python _02_munge_chains.py -d 90 -o "-" -P $1