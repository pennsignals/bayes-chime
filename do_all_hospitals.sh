#!/bin/bash

# for loc in 'CCH' 'LGH' 'PAH' 'Downtown' 'HUP' 'PMC' 'MCP'
# do 
# 	echo $loc
# 	python _01_GOF_sims.py -p data/"$loc"_parameters.csv -t data/"$loc"_ts.csv -C 8 -b -i 5000
# done

for i in '2020_05_08_12_19_54' '2020_05_08_12_15_06' '2020_05_08_12_10_13' '2020_05_08_12_05_17' '2020_05_08_12_00_24' '2020_05_08_11_55_12' '2020_05_08_11_49_38'
do 
	python _02_munge_chains.py -o output/"$i"
done