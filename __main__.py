#!/usr/bin/python3
import csv
import os
import sys

import arg_module
import run_module

sys.path.append(os.getcwd())

def main():
    args = arg_module.parse_args()
    run_module.multi_run(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())
