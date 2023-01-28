#!/bin/sh
mkdir -p /tmp/history/
cp -r *.csv /tmp/history/
.  /cs/usr/benjilieber/PycharmProjects/multi_block_protocol/venv/bin/activate
#module load python3
python3 test.py
deactivate