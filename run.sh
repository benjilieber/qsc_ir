#!/bin/sh
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.0" &
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.0001" &
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.001" &
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.01" &
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.02" &
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.05" &
sbatch -n 11 -t 7-0 --mem=10g --killable single_run.sh "0.1" &
wait