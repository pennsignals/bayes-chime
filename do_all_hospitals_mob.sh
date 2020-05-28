#!/bin/bash

# Set the arguments
chains=16
n_iters=15000
burn_in=10000
reopen_day=100
reopen_speed=.05
reopen_cap=.5


for loc in 'PAH' 'Downtown' 'HUP' 'PMC' 'CCH' 'LGH' 'MCP'
do
	for mob_prior in .2 2
	do
		for beta_prior in .2 2
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
			-o "${loc}_mob${mob_prior}_beta${beta_prior}" \
			--save_chains \
			--include_mobility \
			--override_beta_prior $beta_prior \
			--override_mobility_prior $mob_prior \
			--location_string "$locstring" \
			--ignore_vent 2>> errors.out &
		done
	done
done

# for loc in 'PAH' 'Downtown' 'HUP' 'PMC' 'CCH' 'LGH' 'MCP'
# do

# done







