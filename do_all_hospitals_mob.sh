#!/bin/bash

# Set the arguments
chains=8
n_iters=30000
burn_in=15000
reopen_day=125
reopen_speed=.05
reopen_cap=.5


for loc in 'CCH' 'LGH' 'PAH' 'Downtown' 'HUP' 'PMC' 'MCP'
do
	if [ $loc = 'CCH' ]; then 
		locstring="United States, Pennsylvania, Chester County"
	elif [ $loc = 'LGH' ]; then
		locstring="United States, Pennsylvania, Lancaster County"
	elif [ $loc = 'MCP' ]; then
		locstring="United States, New Jersey, Mercer County"
	else
		locstring="United States, Pennsylvania, Philadelphia County"
	fi
	echo $loc
	echo $chains
	python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv \
	-C $chains \
	-b \
	--penalty .051 \
	-i $n_iters \
	-B $burn_in \
	--reopen_day $reopen_day \
	--reopen_speed $reopen_speed \
	--reopen_cap $reopen_cap \
	--prefix $loc \
	--save_chains \
	-o "${loc}_mob_nobeta_newpriors" \
	--include_mobility \
	--override_beta_prior .001 \
	--override_mobility_prior .1 \
	--location_string "$locstring" \
	--ignore_vent 2>> errors.out &
done

