"""SAEA main"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.Exp import Exp


def main():
    problem = "F12022"
    alg = "SAEA"
    exp = Exp(problem_name=problem, alg_name=alg)
    exp.solve()

main()