

#!/bin/bash

# Set the arguments
chains=8
n_iters=5000
burn_in=2000
reopen_day=100
reopen_speed=.05
reopen_cap=.5 


python _01_GOF_sims.py \
-p data/PAH_parameters.csv \
-t data/PAH_ts.csv \
-C $chains \
-b \
--penalty .06 \
-i $n_iters \
-B $burn_in \
-pp \
--reopen_day $reopen_day \
--reopen_speed $reopen_speed \
--reopen_cap $reopen_cap \
-o "test" \
--include_mobility \
--location_string "United States, Pennsylvania, Philadelphia County"

