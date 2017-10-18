#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:26:32 2017

@author: Valyria
"""
import sklearn
from DFLsklearn import KFold_error 

def interface_funct(model_f,z):
    global X_values,Y_values,n_splits, base_exp,algo, hp_list,has_no_space,var_is_int,space_points,hp_space

    '''
    Some custom model functs, as exmples to implement in the case the user 
    wants to implment some custom interfaces between sklearn and DFL
    '''
    
    
    if(type(model_f) is sklearn.linear_model.ElasticNet):
        model_f.alpha=base_exp**z[0]+base_exp**z[1]
        model_f.l1_ratio=(base_exp**z[0])/(base_exp**z[0]+base_exp**z[1])
        return KFold_error()
    if(type(model_f) is sklearn.linear_model.MultiTaskElasticNet):
        model_f.alpha=base_exp**z[0]+base_exp**z[1]
        model_f.l1_ratio=(base_exp**z[0])/(base_exp**z[0]+base_exp**z[1])
        return KFold_error()

    if(type(model_f) is sklearn.linear_model.BayesianRidge):
        model_f.alpha_1=base_exp**z[0]
        model_f.alpha_2=base_exp**z[1]
        model_f.lambda_1=base_exp**z[2]
        model_f.lambda_2=base_exp**z[3]
        return KFold_error()
    if(type(model_f) is sklearn.linear_model.ARDRegression):
        model_f.alpha_1=base_exp**z[0]
        model_f.alpha_2=base_exp**z[1]
        model_f.lambda_1=base_exp**z[2]
        model_f.lambda_2=base_exp**z[3]
        return KFold_error()


    if(type(model_f) is sklearn.neural_network.MLPRegressor):
        model_f.hidden_layer_sizes=tuple(z[0:len(z)-1].astype(int))
        model_f.alpha=base_exp**z[len(z)-1]
        return KFold_error()
        
        
    if(type(model_f) is sklearn.svm.SVC):
        model_f.C=base_exp**z[0]
        model_f.gamma=base_exp**z[1]
        return KFold_error(errortype="classification")
    
    if(type(model_f) is sklearn.svm.SVR):
        model_f.C=base_exp**z[0]
        model_f.gamma=base_exp**z[1]
        model_f.epsilon=base_exp**z[2]
        return KFold_error()
    
    
    if(type(model_f) is sklearn.svm.LinearSVC):
        model_f.c=base_exp**z[0]
        return KFold_error(errortype="classification")
    if(type(model_f) is sklearn.svm.NuSVC):
        model_f.nu=z[0]
        model_f.gamma=base_exp**z[1]
        return KFold_error(errortype="classification")    


    if(type(model_f) is sklearn.svm.LinearSVR):
        model_f.C=base_exp**z[0]
        return KFold_error()
    if(type(model_f) is sklearn.svm.NuSVR):
        model_f.nu=z[0]
        model_f.gamma=base_exp**z[1]
        model_f.epsilon=base_exp**z[2]
        return KFold_error()
    
    
    if(type(model_f) is sklearn.neural_network.MLPClassifier):
        model_f.hidden_layer_sizes=tuple(z[0:len(z)-1].astype(int))
        model_f.alpha=base_exp**z[len(z)-1]
        return KFold_error(errortype="classification")
    
    
    if(type(model_f) is sklearn.ensemble.RandomForestClassifier):
        model_f.max_features=int(z[0])
        model_f.min_samples_split=int(base_exp**z[1])
        model_f.min_samples_leaf=int(base_exp**z[2])
        model_f.max_depth=int(base_exp**z[3])
        model_f.n_estimators=int(base_exp**z[4])
        return KFold_error(errortype="classification")
    
    if(type(model_f) is sklearn.ensemble.RandomForestRegressor):        
        model_f.max_features=int(z[0])
        model_f.min_samples_split=int(base_exp**z[1])
        model_f.min_samples_leaf=int(base_exp**z[2])
        model_f.max_depth=int(base_exp**z[3])
        model_f.n_estimators=int(base_exp**z[4])
        return set_error()

    if(type(model_f) is sklearn.linear_model.Lasso):
        model_f.alpha=base_exp**z[0]
        return KFold_error()

    if(type(model_f) is R_flarecast_learning_algorithms.R_nn):
        algo.parameters['R_nn']['size']=z[0]
        algo.estimator=R_flarecast_learning_algorithms.R_nn(**algo.parameters)
        model_f=algo.estimator
        return KFold_error()