from algs.differential_evolution.DE_Base import DE_Base


alg_dict = {
    "DE": DE_Base 
}

def get_alg(name):
    return alg_dict[name]
