# -*- coding: utf-8 -*-
"""
Created on Fri Sep 03 09:10:03 2020

@author: Sadaf Farkhani (SF)
"""
import numpy as np
from geoarray import GeoArray
import affine
import pandas as pd
import random
import sys
import os
import gdal
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor as MLP
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
# SF's function
#sys.path.insert(0, 'F:/AU/Phd_courses/DTU_summerSchool/Problem_solving/')
#from img_normalization_sf import normalization


def get_data(prox_pth, orth_pth):
    data = np.load(orth_pth)
    targets = np.load(prox_pth)
    
    return data, targets

def plot_bo(f, bo):
    x = np.linspace(0.00001, 1.5, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(16, 9))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()

def mlp_cv(lr_init, data, targets, n_fold=4):
    """ Utilizing multi-layer perceptron (MLP) accompanied with
    cross validation to optimize the satellite regression.
    Here, initial learning rate value is the only parameter that
    I consider for optimization.
    
    lr_init: the initial learning rate which is selected to be optimized
    data   : RGB-N from the satellite imagery
    targets: proximal grass ratio from rgb images
    """
    estimator = MLP(learning_rate_init=lr_init, activation='relu', solver='adam', alpha=1e-5, batch_size=batch_size, max_iter = 1000, warm_start=True, 
                       learning_rate='adaptive', tol=5*1e-6, random_state=40)
    cval = cross_val_score(estimator, data, targets, cv=n_fold)
    return cval.mean()

def optimize_mlp(data, targets):
    """Apply Bayesian Optimization to MLP parameters."""
    def mlp_crossval(learning_rate_init):
        return mlp_cv(lr_init=learning_rate_init,
                      data=data,
                      targets=targets)
    
    optimizer = BayesianOptimization(
        f=mlp_crossval,
        pbounds={"learning_rate_init": (0.000001, 1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(init_points=10, n_iter=10)

    print("Final result:", optimizer.max)

if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
#        data, targets = get_data()

if __name__ == "__main__":
    data, targets = get_data('F:/AU/Phd_courses/DTU_summerSchool/Problem_solving/data/all_points.npy',
                             'F:/AU/Phd_courses/DTU_summerSchool/Problem_solving/data/data.npy')

    print(Colours.green("--- Optimizing Multi Layer Perceptron ---"))
    optimize_mlp(data, np.squeeze(targets))
