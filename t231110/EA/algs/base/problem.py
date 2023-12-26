import opfunu


def get_problem_by_name(name, ndim=2):
    funcs = opfunu.get_functions_by_classname(name=name)
    return funcs[0](ndim=ndim)

def get_problem_detail(name, ndim=2):
    problem_obj = get_problem_by_name(name, ndim)
    lb = problem_obj._bounds[:, 0]
    ub = problem_obj._bounds[:, 1]
    dim = problem_obj._ndim
    fobj = problem_obj.evaluate

    return lb[0], ub[0], dim, fobj
    