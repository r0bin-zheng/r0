from algs.algorithm_fireworks.FWA_Base import FWA_Base
from algs.algorithm_fireworks.FWA_Impl import FWA_Impl
from algs.differential_evolution.DE_Base import DE_Base
from algs.differential_evolution.DE_Impl import DE_Impl


alg_dict = {
    "DE": DE_Base,
    "DE_Impl": DE_Impl,
    "FWA": FWA_Base,
    "FWA_Impl": FWA_Impl,
}

def get_alg(name):
    return alg_dict[name]
