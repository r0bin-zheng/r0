"""优化算法main"""
from t231110.EA.problem import get_problem_by_name


def run():
    pass


def show():
    # 1. 生成种群迭代gif

    # 2. 生成种群跟踪gif
    pass


def main():
    problem_name = "F12022"
    dims = 2
    method = "GA"

    problem = get_problem_by_name(problem_name, dims)
    alg     = get_algorithm(method)

    res = run(problem, alg)

    show(res, problem, alg)


main()