"""SAEA main"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.Exp import Exp


def main():

    # problem = "Himmelblau" # Himmelblau (2D)
    problem = "Ackley01"
    # problem = "Griewank"
    # problem = "F12022"
    # problem = "Rastrigin"
    # problem = "Ellipsoid"
    # problem = "F62005"
    # problem = "F72005"
    # problem = "Rosenbrock"

    alg = "HSAEA"
    # alg = "DE_SAEA_Base"
    # alg = "FWA_SAEA"
    # alg = "SAEA"

    dim = 2

    fit_max = 200

    surr_types = [
        ["smt_kplsk", "smt_rbf"],
        ["sklearn_gpr"]
    ]

    exp = Exp(problem_name=problem, alg_name=alg, dim=dim, fit_max=fit_max, surr_types=surr_types)
    exp.solve()

main()