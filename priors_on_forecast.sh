

#!/bin/bash

# Set the arguments
chains=8
n_iters=5000
burn_in=2000
reopen_day=100
reopen_speed=.05
reopen_cap=.5 


for fcp in -30. -10. 0. 30. 20.
do 
	echo $fcp
	python _01_GOF_sims.py \
	-p data/Downtown_parameters.csv \
	-t data/Downtown_ts.csv \
	-C $chains \
	-b \
	--penalty .3 \
	-i $n_iters \
	-B $burn_in \
	-pp \
	--reopen_day $reopen_day \
	--reopen_speed $reopen_speed \
	--reopen_cap $reopen_cap \
	--forecast_change_prior_mean " $fcp" \
	--forecast_change_prior_sd 5.0 \
	-o "yprior$fcp"
done

