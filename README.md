
# BayesCHIME

Many factors surrounding the transmission, severity of infections, and remaining susceptibility of local populations to COVID-19 remain highly uncertain. However, as new data on hospitalized cases becomes available, we wish to incorporate this data in order to update and refine our projections of future demand to better inform capacity planners. To that end we have extended CHIME to increase the epidemiological process realism and to coherently incorporate new data as it becomes available. This extension allows us to transition from a small number of scenarios to assess best and worst case projections based on parameter assumptions, to a probabilistic forecast representing a continuous distribution of likely scenarios.


# Installation
We recommend using the [anaconda](https://www.anaconda.com/distribution/) python distribution which comes with most of the libraries you will need to run this application. Additional dependencies should be installed by running:
```bash
pip install -r requirements.txt
```

# Usage

The best way to run sims is via a bash script.  This allows for easy configurability of arguments.  Here is an example script:

```bash

#!/bin/bash

# Set the arguments
chains=8
n_iters=5000
burn_in=2000
reopen_day=100
reopen_speed=.05
reopen_cap=.5 


for loc in 'CCH' 'LGH' 'PAH' 'Downtown' 'HUP' 'PMC' 'MCP'
do 
	echo $loc
	echo $chains
	python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv \
	-C $chains \
	-b \
	-f \
	-i $n_iters \
	-B $burn_in \
	-pp \
	--reopen_day $reopen_day \
	--reopen_speed $reopen_speed \
	--reopen_cap $reopen_cap \
	--prefix $loc
done

```

# Arguments to the command-line interface:

```
usage: _01_GOF_sims.py [-h] [-c MY_CONFIG] [-P PREFIX] [-p PARAMETERS] [-t TS]
                       [-C N_CHAINS] [-i N_ITERS] [-f] [--penalty PENALTY]
                       [-s] [-o OUT] [-a AS_OF] [-b] [-v] [-B BURN_IN]
                       [-d N_DAYS] [-y Y_MAX] [-pp] [--reopen_day REOPEN_DAY]
                       [--reopen_speed REOPEN_SPEED] [--reopen_cap REOPEN_CAP]

Args that start with '--' (eg. -P) can also be set in a config file (specified
via -c). Config file syntax allows: key=value, flag=true, stuff=[a,b,c] (for
details, see syntax at https://goo.gl/R74nmi). If an arg is specified in more
than one place, then commandline values override config file values which
override defaults.

optional arguments:
  -h, --help            show this help message and exit
  -c MY_CONFIG, --my-config MY_CONFIG
                        config file path
  -P PREFIX, --prefix PREFIX
                        prefix for old-style inputs
  -p PARAMETERS, --parameters PARAMETERS
                        the path to the parameters csv
  -t TS, --ts TS        the path to the time-series csv
  -C N_CHAINS, --n_chains N_CHAINS
                        number of chains to run
  -i N_ITERS, --n_iters N_ITERS
                        number of iterations to run per chain
  -f, --fit_penalty     fit the penalty based on the last week of data
  --penalty PENALTY     penalty factor used for shrinkage (0.05 - 1)
  -s, --sample_obs      adds noise to the values in the time-series
  -o OUT, --out OUT     output directory
  -a AS_OF, --as_of AS_OF
                        number of days in the past to project from
  -b, --flexible_beta   flexible, vs simple, logistic represetation of beta
  -v, --verbose         verbose output
  -B BURN_IN, --burn_in BURN_IN
                        how much of the burn-in to discard
  -d N_DAYS, --n_days N_DAYS
                        make a census/admits plot out to n_days
  -y Y_MAX, --y_max Y_MAX
                        max y-scale for the census graph
  -pp, --plot_pairs     Plot posterior samples in a pair-plot grid
  --reopen_day REOPEN_DAY
                        day at which to commence evaluating the reopen
                        function
  --reopen_speed REOPEN_SPEED
                        how fast to reopen
  --reopen_cap REOPEN_CAP
                        how much reopening to allow 0: being fully open, 1: being no change in current social distancing
```
<!-- 
# Usage

First run the sims by passing either the (old-style) prefix or the new-style parameters and ts files.

```bash
# Old-style
python _01_GOF_sims.py -P <prefix> -C <n_chains> -i <n_iters>
# New-style
python _01_GOF_sims.py -p <parameters_file> -t <ts_file> -C <n_chains> -i <n_iters>
```

After the script finishes running it will output the `<output_dir>` which is used by the next step.

To run for each of a list of locations, use:

```bash
for loc in 'CCH' 'LGH' 'Downtown' 'MCP'; do python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv -C 8 -i 5000; done
```

If you want to auto-fit a penalty factor (for shrinkage) you can pass `-f`:
```bash
python _01_GOF_sims.py -p data/$loc_parameters.csv -t data/"$loc"_ts.csv -C 8 -i 5000 -f
```

If you already know the appropriate penalty factor, (and it's common across all the hospitals), do this instead:
```bash
python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv -C 8 -i 5000 --penalty 0.5
```
The penalty factor should be between .05 and less than 1.  1 is maximum penalization.

Results will be saved to `output/<output_dir>/output/chains.json.bz2`, which can be analysed/plotted using:

```bash
python _02_munge_chains.py -o <output_dir>
```

If you would like to run both steps together in a pipeline you can do so using unix pipes:
```bash
python _01_GOF_sims.py -p data/Downtown_parameters.csv -t data/Downtown_ts.csv -o Downtown | python _02_munge_chains.py -o "-" -P Downtown
``` -->

# Docker
For anyone struggling with getting these python scripts running on their own
machine, we've provided a Dockerfile to help you along with the process.
This requires that you setup Docker on your machine. For Mac and Windows
you'll need [Docker Desktop](https://www.docker.com/products/docker-desktop).
On linux you can
[use your package manager](https://runnable.com/docker/install-docker-on-linux)
to install Docker.

## The easy way
The easy way to get things going is through `docker-compose`
### Build the container
`docker-compose build`
### Run the container
`docker-compose up`

By default this will use the CCH files in the `data` directory.
If you'd like to run this with a different set of input files you can override
this using the `$LOC` environment variable. This can be done one of two ways
1. `export LOC=NewLoc`
2. Modify the `.env` file

The `LOC` env variable will be passed to `full_pipe.sh`. If you need to modify
what is passed to the scripts you can modify `full_pipe.sh` accordingly

## The more flexible way
The more flexible way to run the app in Docker is using the `docker` command
### Build the container
`docker build -t chime_sims .`
### Run the container
``docker run -it --rm -v `pwd`:/chime_sims chime_sims /bin/bash``

*Note: You may have to replace `` `pwd` `` with the path to the chime_sims
directory.*

To run the container in Docker on Windows, you must enable sharing from the local drive where the chime_sims
directory exists. The following steps are for Windows 10 and Docker Desktop version 2.2.0.5.
1. Right click on the Docker icon in the system tray and select settings.
2. Click Resources and navigate to File Sharing.
3. Select the checkbox for the appropriate local drive.
4. Click Apply & Restart (you may be prompted to enter your credentials).
To run the container, insert the file path to the chime_sims directory on your local drive:
`docker run -it --rm -v InsertFilePath:/chime_sims chime_sims /bin/bash`

### Run the scripts
Once the container starts you should be put into a bash shell where you can
then run scripts as described above. 

<!-- For example:

`python _01_GOF_sims.py -p data/Downtown_parameters.csv -t data/Downtown_ts.csv -o Downtown | python _02_munge_chains.py -o "-" -P Downtown` -->

*Note: You may need to increase resources to run the scripts. To do so, right click on the Docker icon in the
system tray and select settings then click Resources and adjust as needed.*
