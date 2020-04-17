
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

