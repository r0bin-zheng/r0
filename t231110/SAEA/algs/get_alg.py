import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.fwa_surr.FWA_SAEA import FWA_SAEA
from SAEA.algs.de_surr.DE_SAEA_Base import DE_SAEA_Base
from SAEA.algs.base.HSAEA_Base import HSAEA_Base
from SAEA.algs.fd_hsaea.FD_HSAEA_Base import FD_HSAEA_Base
from SAEA.algs.ga_surr.GA_SAEA_Base import GA_SAEA_Base
from SAEA.algs.pso_surr.PSO_SAEA_Base import PSO_SAEA_Base
from SAEA.algs.pd_hsaea.PD_HSAEA_Base import PD_HSAEA_Base
from SAEA.algs.dd_hsaea.DD_HSAEA_Base import DD_HSAEA_Base
from SAEA.algs.gd_hsaea.GD_HSAEA_Base import GD_HSAEA_Base


alg_dict = {
    "SAEA": SAEA_Base,
    "FWA_SAEA": FWA_SAEA,
    "DE_SAEA_Base": DE_SAEA_Base,
    "GA_SAEA_Base": GA_SAEA_Base,
    "PSO_SAEA_Base": PSO_SAEA_Base,
    
    "HSAEA": HSAEA_Base,
    "FD_HSAEA": FD_HSAEA_Base,
    "PD_HSAEA": PD_HSAEA_Base,
    "DD_HSAEA": DD_HSAEA_Base,
    "GD_HSAEA": GD_HSAEA_Base,
}

def get_alg(name):
    return alg_dict[name]