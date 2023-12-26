import opfunu


def get_problem_by_name(name, ndim=2):
    funcs = opfunu.get_functions_by_classname(name=name)
    return funcs[0](ndim=ndim)
