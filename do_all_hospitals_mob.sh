#!/bin/bash

# Set the arguments
chains=8
n_iters=15000
burn_in=10000
reopen_day=100
reopen_speed=.05
reopen_cap=.2

# Fit flexible beta, with shrinkage specified to be .25 (small)
for loc in 'PAH' 'Downtown' 'HUP' 'PMC'
do 
	echo $loc
	echo $chains
	python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv \
	-C $chains \
	-b \
	--penalty .25 \
	-i $n_iters \
	-B $burn_in \
	-pp \
	--reopen_day $reopen_day \
	--reopen_speed $reopen_speed \
	--reopen_cap $reopen_cap \
	--prefix $loc \
	-o "${loc}_mob" \
	--save_chains \
	--include_mobility \
	--location_string "United States, Pennsylvania, Philadelphia County" \
	--ignore_vent 2>> errors.out &
done
