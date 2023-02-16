#!/usr/bin/env python3
"""
runs gurobi in solution pool mode to find many solutions
"""

from gurobipy import *
from argparse import ArgumentParser
from os import path, mkdir
import re
from ast import literal_eval
from itertools import product

from skinny_milp import tikzify_milp_wrapper

def get_num_rounds(model: Model):
    numrounds = 0
    for var in model.getVars():
        name = var.VarName
        m = re.match(r'^sbox\[([0-3]),([0-3]),(\d+)\]$', name)
        if not m:
            continue

        rnd = int(m[3])
        numrounds = max(numrounds, rnd)
    return numrounds


def read_sol_file(fname):
    res = dict()
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            name, val = line.rsplit(maxsplit=1)
            res[name] = literal_eval(val)
    return res

def set_start_solution(fname, model):
    sol = read_sol_file(fname)
    for var in model.getVars():
        if var.VarName in sol:
            var.Start = sol[var.VarName]


def restrict_external_solutions(m: Model, numrounds: int):
    for rnd, row, col in product(range(numrounds + 1), range(4), range(4)):
        sbox = m.getVarByName(f'sbox[{row},{col},{rnd}]')
        tbox = m.getVarByName(f'tbox[{row},{col},{rnd}]')
        unkn = m.getVarByName(f'unkn[{row},{col},{rnd}]')
        forg = m.getVarByName(f'forg[{row},{col},{rnd}]')

        m.addConstr(unkn <= tbox)

        #from IPython import embed
        #embed()
        #sys.exit(0)
        #pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename', type=str, help='input file in .lp/mps file format')
    parser.add_argument('-n', '--poolsize', type=int, default=10, help='number of solutions to search for')
    parser.add_argument('-m', '--mode', type=int, default=2, choices=[0,1,2], help='https://www.gurobi.com/documentation/9.5/refman/poolsearchmode.html')
    args = parser.parse_args()

    m = read(args.filename)
    m.Params.PoolSolutions = args.poolsize
    m.Params.PoolSearchMode = args.mode

    numrounds = get_num_rounds(m)

    out_dir, _ = path.splitext(args.filename)
    base_name = path.basename(out_dir)
    sol_file = out_dir + '.sol'

    try:
        #set_start_solution(sol_file, m)
        ...
    except FileNotFoundError:
        print(f"could not read starting solution from '{sol_file}'")
        sys.exit(1)

    restrict_external_solutions(m, numrounds)

    mkdir(out_dir)
    m.optimize()
    for m.Params.SolutionNumber in range(m.SolCount):
        fname = path.join(out_dir, f'{base_name}_{m.Params.SolutionNumber}')
        m.write(f'{fname}.sol')
        tikzify_milp_wrapper(fname, numrounds, m)
