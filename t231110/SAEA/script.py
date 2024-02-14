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
args = parser.parse_args()

problem = args.problem
alg = args.alg
id = args.id

exp = Exp(problem_name=problem, alg_name=alg, id=id)
exp.solve()
