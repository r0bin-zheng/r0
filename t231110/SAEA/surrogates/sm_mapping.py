"""字符串和代理模型类的映射"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.surrogates.smt_sm import SmtKriging, SmtKPLSK, SmtRBF
from SAEA.surrogates.sklearn_sm import SklearnGaussianProcessRegressor

SM_DICT = {
    "smt_kriging": SmtKriging,
    "smt_kplsk": SmtKPLSK,
    "smt_rbf": SmtRBF,
    "sklearn_gpr": SklearnGaussianProcessRegressor,
}