from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem, Problem
from scipy.stats import norm
from pymoo.algorithms.soo.nonconvex.ga import GA

# EGO算法
# 使用conda环境moo1
# pymoo的问题基类

# Objective function (replace with your expensive function)
def objective_function(x):
    return (6*x - 2)**2 * np.sin(12*x - 4)

# Create an initial design of experiments using Latin Hypercube Sampling
xlimits = np.array([[0.0, 1.0]])
sampling = LHS(xlimits=xlimits)
num_initial_points = 10
X_train = sampling(num_initial_points)
y_train = objective_function(X_train)

# Train the surrogate model using a Kriging model
krg = KRG(theta0=[1e-2], 
          print_training=False,
          print_prediction=False, 
          print_problem=False,
          print_solver=False)
krg.set_training_values(X_train, y_train)
krg.train()

# Acquisition function: Expected Improvement
def expected_improvement(X, krg, y_best):
    y_pred, y_std = krg.predict_values(X), np.sqrt(krg.predict_variances(X))
    improvement = y_best - y_pred
    Z = improvement / y_std
    EI = improvement * norm.cdf(Z) + y_std * norm.pdf(Z)
    return EI

# Define the problem for pymoo's minimize method
class AcquisitionProblem(ElementwiseProblem):
    def __init__(self, krg, y_best):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=0.0, xu=1.0)
        self.krg = krg
        self.y_best = y_best
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -expected_improvement(np.array([x]), self.krg, self.y_best)

# Define the problem for pymoo's minimize method
class EGOProblem(Problem):
    def __init__(self, krg, y_best):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=np.array([0]), xu=np.array([1]))
        self.krg = krg
        self.y_best = y_best

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -expected_improvement(x[0], self.krg, self.y_best)

# Optimization loop
n_iter = 20
for i in range(n_iter):
    # Find the current best solution
    y_best = np.min(y_train)

    # Define the optimization problem
    problem = EGOProblem(krg, y_best)

    # Use 'pymoo' to minimize the acquisition function
    res = minimize(problem,
                   method='nelder-mead',
                   options={'x0': np.array([0.5]), 'bounds': np.array([[0., 1.]])})

    # Get the new x
    new_x = res.X[0]

    # Get the new y
    new_y = objective_function(new_x)

    # Update the training set
    X_train = np.vstack((X_train, new_x))
    y_train = np.append(y_train, new_y)

    # Retrain the surrogate model
    krg.set_training_values(X_train, y_train)
    krg.train()

# Find the best point found during the optimization
best_index = np.argmin(y_train)
best_point = X_train[best_index]
best_value = y_train[best_index]

print('Best point:', best_point)
print('Best objective function value:', best_value)
