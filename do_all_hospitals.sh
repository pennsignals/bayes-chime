#!/bin/bash

# Set the arguments
chains=8
n_iters=5000
burn_in=2000
reopen_day=100
reopen_speed=.05
reopen_cap=.2

# Fit flexible beta, with shrinkage specified to be .25 (small)
for loc in 'CCH' 'LGH' 'PAH' 'Downtown' 'HUP' 'PMC' 'MCP'
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
	-o "${loc}_flexB_novent" \
	--ignore_vent 2>> errors.out &
done

# fit logistic as a backup
for loc in 'CCH' 'LGH' 'PAH' 'Downtown' 'HUP' 'PMC' 'MCP'
do 
	echo $loc
	echo $chains
	python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv \
	-C $chains \
	-i $n_iters \
	-B $burn_in \
	-pp \
	--reopen_day $reopen_day \
	--reopen_speed $reopen_speed \
	--reopen_cap $reopen_cap \
	--prefix $loc \
	-o "${loc}_logistic_novent" \
	--ignore_vent 2>> errors.out &
done

# fit a version for LGH and CCH that has a downward prior, reflecting the fact that we know that there were clusters of LTC cases that won't get replicated next week:
# Prior:  going down by 10%, plus or minus 5%
for loc in 'CCH' 'LGH' 'PAH' 'Downtown' 'HUP' 'PMC' 'MCP'
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
	-o "${loc}_downward_prior_novent" \
	--forecast_change_prior_mean " -10" \
	--forecast_change_prior_sd " 5" \
	--ignore_vent 2>> errors.out &
done



