from algs.algorithm_fireworks.FWA_Base import FWA_Base
from algs.differential_evolution.DE_Base import DE_Base


alg_dict = {
    "DE": DE_Base,
    "FWA": FWA_Base,
}

def get_alg(name):
    return alg_dict[name]
