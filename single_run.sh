#!/bin/sh
#SBATCH --output="$1,p_err=$2.out"
.  /cs/usr/benjilieber/PycharmProjects/qsc_ir/venv/bin/activate
python3 test.py "$1" "$2"
deactivate