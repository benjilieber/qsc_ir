#!/bin/sh
mkdir -p /tmp/history/
mkdir -p /tmp/history/0.0/
mkdir -p /tmp/history/0.0001/
mkdir -p /tmp/history/0.001/
mkdir -p /tmp/history/0.01/
mkdir -p /tmp/history/0.02/
mkdir -p /tmp/history/0.05/
mkdir -p /tmp/history/0.1/
cp -r slurm-13909851.out /tmp/history/0.0/  # 0.0
cp -r slurm-13986072.out /tmp/history/0.0/  # 0.0
cp -r slurm-13992471.out /tmp/history/0.0/  # 0.0
cp -r slurm-14024353.out /tmp/history/0.0/  # 0.0
cp -r slurm-14048189.out /tmp/history/0.0/  # 0.0
cp -r slurm-13909852.out /tmp/history/0.0001/  # 0.0001
cp -r slurm-13986073.out /tmp/history/0.0001/  # 0.0001
cp -r slurm-14027574.out /tmp/history/0.0001/  # 0.0001
# cp -r slurm-?.out /tmp/history/0.0001/  # 0.0001
cp -r slurm-13909853.out /tmp/history/0.001/  # 0.001
cp -r slurm-13986074.out /tmp/history/0.001/  # 0.001
cp -r slurm-14027578.out /tmp/history/0.001/  # 0.001
# cp -r slurm-?.out /tmp/history/0.001/  # 0.001
cp -r slurm-13909854.out /tmp/history/0.01/  # 0.01
cp -r slurm-13986077.out /tmp/history/0.01/  # 0.01
cp -r slurm-14027579.out /tmp/history/0.01/  # 0.01
# cp -r slurm-?.out /tmp/history/0.01/  # 0.01
cp -r slurm-13909855.out /tmp/history/0.02/  # 0.02
cp -r slurm-13986078.out /tmp/history/0.02/  # 0.02
cp -r slurm-14027580.out /tmp/history/0.02/  # 0.02
# cp -r slurm-?.out /tmp/history/0.02/  # 0.02
cp -r slurm-13909856.out /tmp/history/0.05/  # 0.05
cp -r slurm-13986079.out /tmp/history/0.05/  # 0.05
cp -r slurm-14027582.out /tmp/history/0.05/  # 0.05
# cp -r slurm-?.out /tmp/history/0.05/  # 0.05
cp -r slurm-13909857.out /tmp/history/0.1/  # 0.1
cp -r slurm-13986080.out /tmp/history/0.1/  # 0.1
cp -r slurm-13992491.out /tmp/history/0.1/  # 0.1
cp -r slurm-14024760.out /tmp/history/0.1/  # 0.1
cp -r slurm-14048169.out /tmp/history/0.1/  # 0.1
#cp -r *.csv /tmp/history/
.  /cs/usr/benjilieber/PycharmProjects/multi_block_protocol/venv/bin/activate
#module load python3
python3 test.py
deactivate
