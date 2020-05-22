

#!/bin/bash

# Set the arguments
chains=8
n_iters=5000
burn_in=2000
reopen_day=100
reopen_speed=.05
# reopen_cap=.5 

for reopen_cap in 0.0 .2 .8 1
do
	python _01_GOF_sims.py \
	-p data/CCH_parameters.csv \
	-t data/CCH_ts.csv \
	-C $chains \
	-b \
	-i $n_iters \
	-B $burn_in \
	-pp \
	--reopen_day $reopen_day \
	--reopen_speed $reopen_speed \
	--reopen_cap $reopen_cap \
	-o "${reopen_cap}test" \
	--ignore_vent &
done
