import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.fwa_surr.FWA_SAEA import FWA_SAEA

alg_dict = {
    "SAEA": SAEA_Base,
    "FWA_SAEA": FWA_SAEA,
}

def get_alg(name):
    return alg_dict[name]