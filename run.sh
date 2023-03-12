#!/bin/sh

p_err="0.1"
mkdir -p /tmp/history/
case $p_err in
   "0.0")
     mkdir -p /tmp/history/0.0/
     cp -r slurm-14410939.out /tmp/history/0.0/
   ;;
   "0.0001")
     mkdir -p /tmp/history/0.0001/
     cp -r slurm-14410940.out /tmp/history/0.0001/
   ;;
   "0.001")
     mkdir -p /tmp/history/0.001/
     cp -r slurm-14410941.out /tmp/history/0.001/
   ;;
   "0.01")
     mkdir -p /tmp/history/0.01/
     cp -r slurm-14410942.out /tmp/history/0.01/
   ;;
   "0.02")
     mkdir -p /tmp/history/0.02/
     cp -r slurm-14410943.out /tmp/history/0.02/
   ;;
   "0.05")
     mkdir -p /tmp/history/0.05/
     cp -r slurm-14410944.out /tmp/history/0.05/
   ;;
   "0.1")
     mkdir -p /tmp/history/0.1/
     cp -r slurm-14410945.out /tmp/history/0.1/
   ;;
esac
#cp -r *.csv /tmp/history/
.  /cs/usr/benjilieber/PycharmProjects/multi_block_protocol/venv/bin/activate
#module load python3
python3 test.py $p_err
deactivate
