# Usage

First run the sims by passing the name of the location (matching the prefix of the `data/<prefix>_parameters.csv` and `data/<prefix>_ts.csv` files).

```bash
python _01_GOF_sims.py <prefix> <n_chains> <n_iters><penalty_factor>
```

To run for each of a list of locations, use:

```bash
for loc in 'CCH' 'LGH' 'Downtown' 'MCP'; do python _01_GOF_sims.py $loc 8 5000 -99; done
```

If you already know the appropriate penalty factor, (and it's common across all the hospitals), do this instead:

```bash
for loc in 'CCH' 'LGH' 'Downtown' 'MCP'; do python _01_GOF_sims.py $loc 8 5000 <<known_penalty_factor>>; done
```
THe penalty factor should be between .05 and less than 1.  1 is maximum penalization.

Results will be saved to `output/<prefix>_chains.pkl`, which can be analysed/plotted using:

```bash
for loc in 'CCH' 'LGH' 'Downtown' 'MCP'; do python _02_munge_chains.py $loc; done
```
