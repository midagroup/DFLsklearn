#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:46:41 2017

@authors: V. Latorre, F. Benvenuto
"""
import numpy
import sklearn
import DFLsklearn 

def demo_classification():
#------------------------------------------------------------------------------
#Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    
    print('Loading the iris dataset from sklearn')
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
#Create the DFLsklearn object and train
#------------------------------------------------------------------------------
    print('Initializing and training the MLPClassifier')
    estimator_path='sklearn.neural_network.MLPClassifier'
    model=DFLsklearn.DFL_estimator(estimator_path=estimator_path,
                                  estimator_param={'random_state':0},
                                  metric='accuracy_score',
                                  minimization=False,
                                  iprint=0)
    model.fit(X_train,Y_train)
    
    print('Optimization complete, accuracy score on the test set: %f'
          % sklearn.metrics.accuracy_score(model.predict(X_test),Y_test))




def demo_regression():
#------------------------------------------------------------------------------
    #Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    
    print('Loading the diabetes dataset from sklearn')
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
#Create the DFLsklearn object and train
#------------------------------------------------------------------------------ 
    
    print('Initializing and training the MLPClassifier')    
    estimator_path='sklearn.neural_network.MLPRegressor'
    model=DFLsklearn.DFL_estimator(estimator_path=estimator_path,
                                  estimator_param={'random_state':0},
                                  metric='r2_score',
                                  minimization=False,
                                  iprint=0)
    model.fit(X_train,Y_train)

    print('Optimization complete, coef. of determination on the test set: %f'
          % sklearn.metrics.r2_score(model.predict(X_test),Y_test))
            
    
def demo_custom():
    '''
    Custom hyperparameters optimization, by setting the formal paramters for a
    support vector machine    
    '''
#------------------------------------------------------------------------------
#Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    
    print('Loading the diabetes dataset from sklearn')
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
#Create the DFLsklearn object and train
#------------------------------------------------------------------------------
    
    print('Initializing and training the MLPClassifier')  
    estimator_path='sklearn.neural_network.MLPRegressor'
    model=DFLsklearn.DFL_estimator(estimator_path=estimator_path,
                                  preset_config=0,
                                  base=10.0,
                                  hp_list=['hidden_layer_sizes','alpha'], 
                                  hp_init=[(10,10,),-8], 
                                  lb=[10,10,-8], 
                                  ub=[100,100,8], 
                                  step=[1,1,1], 
                                  init_int_step=[10,10,1],
                                  is_integer=[1,1,0],
                                  on_a_mesh=[0,0,1],
                                  var_is_int=[0,0,0],
                                  estimator_param={'random_state':1},
                                  iprint=0)    
    model.fit(X_train,Y_train)

    print('Optimization complete, coef. of determination on the test set: %f'
          % sklearn.metrics.r2_score(model.predict(X_test),Y_test))
        
def demo_custom_preset():
    '''
    Custom hyperparameters optimization, by setting the formal paramters for a
    support vector machine    
    '''
#------------------------------------------------------------------------------
#Load the dateset and split it in training and test sets
#------------------------------------------------------------------------------
    
    print('Loading the diabetes dataset from sklearn')
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
#Create the DFLsklearn object and train
#------------------------------------------------------------------------------
    
    print('Initializing and training the MLPClassifier')  
    estimator_path='sklearn.neural_network.MLPRegressor'
    model=DFLsklearn.DFL_estimator(estimator_path=estimator_path,
                                  preset_function=newpreset,
                                  estimator_param={'random_state':1})
    
    model.fit(X_train,Y_train)

    print('Optimization complete, coef. of determination on the test set: %f'
          % sklearn.metrics.r2_score(model.predict(X_test),Y_test))
        

def newpreset(model_f):
    n=3
    base=10.0
    hp_list=['hidden_layer_sizes','alpha']
    hp_init=[(10,10,),-8]
    lb=[10,10,-8]
    ub=[100,100,8]
    step=[1,1,1]
    init_int_step=[10,10,1]
    is_integer=[1,1,0]
    on_a_mesh=[0,0,1]
    var_is_int=[0,0,0]

    return n,hp_init,lb,ub,step,init_int_step,is_integer,base,hp_list,on_a_mesh,var_is_int




if __name__ == '__main__':
    demo_regression()
