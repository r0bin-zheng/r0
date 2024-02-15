import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.fwa_surr.FWA_SAEA import FWA_SAEA
from SAEA.algs.de_surr.DE_SAEA_Base import DE_SAEA_Base
from SAEA.algs.base.HSAEA_Base import HSAEA_Base
from SAEA.algs.fd_hsaea.FD_HSAEA_Base import FD_HSAEA_Base


alg_dict = {
    "SAEA": SAEA_Base,
    "HSAEA": HSAEA_Base,
    "FD_HSAEA": FD_HSAEA_Base,
    "FWA_SAEA": FWA_SAEA,
    "DE_SAEA_Base": DE_SAEA_Base,
}

def get_alg(name):
    return alg_dict[name]