#!/bin/sh
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.0" "polar" &
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.0001" "polar" &
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.001" "polar" &
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.01" "polar" &
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.02" "polar" &
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.05" "polar" &
sbatch -n 101 -t 7-0 --mem=10g --killable --requeue single_run.sh "0.1" "polar" &
wait