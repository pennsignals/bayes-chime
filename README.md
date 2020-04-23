
# BayesCHIME

Many factors surrounding the transmission, severity of infections, and remaining susceptibility of local populations to COVID-19 remain highly uncertain. However, as new data on hospitalized cases becomes available, we wish to incorporate this data in order to update and refine our projections of future demand to better inform capacity planners. To that end we have extended CHIME to increase the epidemiological process realism and to coherently incorporate new data as it becomes available. This extension allows us to transition from a small number of scenarios to assess best and worst case projections based on parameter assumptions, to a probabilistic forecast representing a continuous distribution of likely scenarios.


# Installation
We recommend using the [anaconda](https://www.anaconda.com/distribution/) python distribution which comes with most of the libraries you will need to run this application. Additional dependencies should be installed by running:
```bash
pip install -r requirements.txt
```

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
```

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

###Run the container on Windows
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
then run scripts as described above. For example:

`python _01_GOF_sims.py -p data/Downtown_parameters.csv -t data/Downtown_ts.csv -o Downtown | python _02_munge_chains.py -o "-" -P Downtown`

*Note: You may need to increase resources to run the scripts. To do so, right click on the Docker icon in the
system tray and select settings then click Resources and adjust as needed.*
