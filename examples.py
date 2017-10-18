#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:46:41 2017

@author: Valyria
"""
import numpy
import sklearn
import DFLscikit

def demo_classification():
#------------------------------------------------------------------------------
#Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    
    dataset=sklearn.datasets.load_iris()    
    X=dataset.data
    Y=dataset.target    
    seed=1
    test_size = int( 0.2 * len( Y ) )
    numpy.random.seed( seed )
    indices = numpy.random.permutation(len(X))
    X_train = X[ indices[:-test_size]]
    Y_train = Y[ indices[:-test_size]]
    X_test = X[ indices[-test_size:]]
    Y_test = Y[ indices[-test_size:]]
       
#------------------------------------------------------------------------------
#Create the estimator from the libraries of sklearn
#------------------------------------------------------------------------------
    estimator=sklearn.neural_network.MLPClassifier(random_state=0)
#------------------------------------------------------------------------------
#Incapsulate the estimator inside of DFLscikit and train
#------------------------------------------------------------------------------
    
    model=DFLscikit(estimator,loss='accuracy_score',minimization=False)
    model.fit(X_train,Y_train)
    
    print(sklearn.metrics.accuracy_score(model.predict(X_test),Y_test))




def demo_regression():
#------------------------------------------------------------------------------
    #Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    dataset=sklearn.datasets.load_diabetes()        
    X=dataset.data
    Y=dataset.target        
    seed=1
    test_size = int( 0.2 * len( Y ) )
    numpy.random.seed( seed )
    indices = numpy.random.permutation(len(X))
    X_train = X[ indices[:-test_size]]
    Y_train = Y[ indices[:-test_size]]
    X_test = X[ indices[-test_size:]]
    Y_test = Y[ indices[-test_size:]]

#------------------------------------------------------------------------------
#Create the estimator from the libraries of sklearn
#------------------------------------------------------------------------------
    estimator=sklearn.neural_network.MLPRegressor(random_state=0)
#------------------------------------------------------------------------------
#Incapsulate the estimator inside of DFLscikit and train
#------------------------------------------------------------------------------
    model=DFLscikit(estimator,loss='r2_score',minimization=False)
    model.fit(X_train,Y_train)
        
    print(sklearn.metrics.r2_score(model.predict(X_test),Y_test))
    
    
def Custom_search():
    '''
    Custom hyperparameters optimization, by setting the formal paramters for a
    support vector machine    
    '''
#------------------------------------------------------------------------------
    #Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    dataset=sklearn.datasets.load_diabetes()        
    X=dataset.data
    Y=dataset.target        
    seed=1
    test_size = int( 0.2 * len( Y ) )
    numpy.random.seed( seed )
    indices = numpy.random.permutation(len(X))
    X_train = X[ indices[:-test_size]]
    Y_train = Y[ indices[:-test_size]]
    X_test = X[ indices[-test_size:]]
    Y_test = Y[ indices[-test_size:]]

#------------------------------------------------------------------------------
#Create the estimator from the libraries of sklearn, to its default values
#------------------------------------------------------------------------------
    estimator=sklearn.neural_network.MLPRegressor(random_state=0)
    model=DFLscikit(estimator,custom=1,n=3,loss='r2_score',minimization=False,hp_list=['C','gamma','epsilon'],has_no_space=[0,0,0],var_is_int=numpy.zeros(3), hp_space=['default','default','default'],base=2,z0=numpy.array([2,2,0.5]), lb=numpy.array([-5,-15,-7]), ub=numpy.array([15,3,3]), step=numpy.array([1]*3), init_int_step=numpy.array([1]*3),is_integer=numpy.ones(3))
    model.fit(X_train,Y_train)
        
    print(sklearn.metrics.r2_score(model.predict(X_test),Y_test))
