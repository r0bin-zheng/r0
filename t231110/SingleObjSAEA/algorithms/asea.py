# assemble surrogate model assisted evolutionary algorithm
import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.spx import SPX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.algorithms.soo.nonconvex.ga import GA

class ASEA1(GA):

    def __init__(self,
                 pop_size=100,
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 **kwargs):
        super.__init__(pop_size=pop_size,
                    eliminate_duplicates=eliminate_duplicates,
                    n_offsprings=n_offsprings,
                    **kwargs)
    
    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)