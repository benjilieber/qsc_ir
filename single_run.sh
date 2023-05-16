#!/bin/sh
.  /cs/labs/benor/benjilieber/venv/bin/activate
python3 test.py "$1" "$2"
deactivate