"""SAEA main"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import argparse
from SAEA.Exp import Exp
from SAEA.utils.common1 import get_id

parser = argparse.ArgumentParser(description="")
parser.add_argument("--problem", "-p", type=str, default="Ackley01")
parser.add_argument("--alg", "-a", type=str, default="HSAEA")
parser.add_argument("--id", "-i", type=str, default=get_id())
parser.add_argument("--dim", "-d", type=int, default=2)
parser.add_argument("--fit_max", "-m", type=int, default=200)
parser.add_argument("--selection_strategy", "-s", type=str, default="MixedNonlinearly")
args = parser.parse_args()

problem = args.problem
alg = args.alg
id = args.id
dim = int(args.dim)
fit_max = args.fit_max
selection_strategy = args.selection_strategy

exp = Exp(problem_name=problem, alg_name=alg, id=id, dim=dim, 
          fit_max=fit_max, selection_strategy=selection_strategy,
          ss_args=None)
exp.solve()
