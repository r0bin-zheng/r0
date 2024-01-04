import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.SAEA_Base import SAEA_Base

alg_dict = {
    "SAEA": SAEA_Base,
}

def get_alg(name):
    return alg_dict[name]