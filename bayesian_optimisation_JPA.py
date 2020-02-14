from emukit.core import ParameterSpace, ContinuousParameter
from emukit.experimental_design.model_free.random_design import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
import numpy as np
import GPy
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import json

# Decision loops
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
# Acquisition functions
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
# Acquistions optimizers
from emukit.core.optimization import GradientAcquisitionOptimizer
# Stopping conditions
from emukit.core.loop import FixedIterationsStoppingCondition
# Point calculator
from emukit.core.loop import SequentialPointCalculator
# Bayesian quadrature kernel and model
from emukit.quadrature.kernels import QuadratureRBF

# COMSOL simulation
from functions import get_SNR


def run_optimisation(current_range,freq_range,power_range):
    parameter_space = ParameterSpace([\
        ContinuousParameter('current', current_range[0], current_range[1]), \
        ContinuousParameter('freq', freq_range[0], freq_range[1]), \
        ContinuousParameter('power', power_range[0], power_range[1])
        ])

    def function(X):
        current = X[:,0]
        freq = X[:,1]
        power = X[:,2]
        out = np.zeros((len(current),1))
        for g in range(len(current)):
            '''
            Set JPA Current, Frequency & Power
            '''
            out[g,0] = -get_SNR(plot=False)[-1] #Negative as want to maximise SNR
        return out


    num_data_points = 10

    design = RandomDesign(parameter_space)
    X = design.get_samples(num_data_points)
    Y = function(X)

    model_gpy = GPRegression(X,Y)
    model_gpy.optimize()
    model_emukit = GPyModelWrapper(model_gpy)

    exp_imprv = ExpectedImprovement(model = model_emukit)
    optimizer = GradientAcquisitionOptimizer(space = parameter_space)
    point_calc = SequentialPointCalculator(exp_imprv,optimizer)
    coords = []
    min = []

    bayesopt_loop = BayesianOptimizationLoop(model = model_emukit,
                                             space = parameter_space,
                                             acquisition=exp_imprv,
                                             batch_size=1)

    stopping_condition = FixedIterationsStoppingCondition(i_max = 100)

    bayesopt_loop.run_loop(q, stopping_condition)


    coord_results  = bayesopt_loop.get_results().minimum_location
    min_value = bayesopt_loop.get_results().minimum_value
    step_results = bayesopt_loop.get_results().best_found_value_per_iteration
    print(coord_results)
    print(min_value)

    return coord_results,abs(min_value)
