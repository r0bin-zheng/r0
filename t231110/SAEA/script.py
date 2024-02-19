"""SAEA main"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import argparse
import json
from SAEA.Exp import Exp
from SAEA.utils.common1 import get_id

parser = argparse.ArgumentParser(description="")
parser.add_argument("--problem", "-p", type=str, default="Ackley01")
parser.add_argument("--alg", "-a", type=str, default="HSAEA")
parser.add_argument("--id", "-i", type=str, default=get_id())
parser.add_argument("--dim", "-d", type=int, default=2)
parser.add_argument("--fit_max", "-m", type=int, default=200)
parser.add_argument("--surr", "-sm", type=str, default="""[["smt_kplsk", "smt_rbf"], ["sklearn_gpr"]]""")
parser.add_argument("--selection_strategy", "-s", type=str, default="MixedNonlinearly")
parser.add_argument("--fwa_wt_strategy", "-fws", type=str, default="NegativeCorrelation")
args = parser.parse_args()

problem = args.problem
alg = args.alg
id = args.id
dim = int(args.dim)
fit_max = args.fit_max
# surr_str = args.surr.replace("__spacem3__", " ")
# print(surr_str)
surr = args.surr
selection_strategy = args.selection_strategy
fwa_wt_strategy = args.fwa_wt_strategy

def get_surr_types(surr_str):
    surr_types = []
    surr_list = surr_str.split("@")
    print(surr_list)
    for s in surr_list:
        surr_types.append(s.split("+"))
    return surr_types

surr_types = get_surr_types(surr)
print(surr_types)
exp = Exp(debug=False, problem_name=problem, alg_name=alg, id=id, dim=dim, fit_max=fit_max, 
          surr_types=surr_types, selection_strategy=selection_strategy,
          ss_args=None, fws=fwa_wt_strategy)
exp.solve()
