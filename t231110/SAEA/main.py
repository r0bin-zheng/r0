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

    # alg = "HSAEA"
    # alg = "DE_SAEA_Base"
    # alg = "FWA_SAEA"
    # alg = "SAEA"
    # alg = "GA_SAEA_Base"
    # alg = "PSO_SAEA_Base"
    # alg = "FD_HSAEA"
    # alg = "DD_HSAEA"
    alg = "GD_HSAEA"

    dim = 2

    fit_max = 200

    # 此项只有在SAEA和HSAEA作为alg时有效
    ea_types = [
        "FWA_Surr_Impl2",
        # "DE_Surr_Base",
        # "PSO_Surr_Base",
        "DE_Surr_Base",
    ]

    surr_types = [
        ["smt_kplsk", "smt_rbf"],
        ["smt_kplsk"]
    ]

    debug = False

    exp = Exp(debug=debug, problem_name=problem, alg_name=alg, dim=dim, 
              fit_max=fit_max, ea_types=ea_types, surr_types=surr_types)
    exp.solve()

main()