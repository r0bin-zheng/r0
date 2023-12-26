import opfunu

funcs1 = opfunu.get_functions_by_classname("F12022")
funcs2 = opfunu.get_functions_by_classname("F22022")
funcs3 = opfunu.get_functions_by_classname("F32022")
funcs4 = opfunu.get_functions_by_classname("F42022")
funcs5 = opfunu.get_functions_by_classname("F52022")

func = funcs1[0](ndim=10)
func.evaluate(func.create_solution())

all_funcs_2014 = opfunu.get_functions_based_classname("2014")
print(all_funcs_2014)