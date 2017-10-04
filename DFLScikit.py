#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:55:44 2017

@author: Valyria
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:19:22 2017

@author: Valyria
"""
#import sklearn.datasets
#set=sklearn.datasets.load_boston()
#X=set.data
#Y=set.target
#f2py -c dfl.pyf sd.f90 main_box.f90
import numpy 
import sklearn.linear_model
import dfl
from sklearn.model_selection import KFold
import sklearn.datasets
import sklearn.neural_network
import sklearn.svm
import sklearn.ensemble
from sklearn.base import BaseEstimator


def log_base(x,base):
    return numpy.log(x)/numpy.log(base)


def start_up_interface(model_f,hp_list,has_no_space,var_is_int):
    global X_values
    '''
    Method that automatically initializes some of the estimators of 
    scikit-learn for a more user-friendly interface. The currenty supported 
    estimators are:
        sklearn.linear_model.ElasticNet
        sklearn.linear_model.MultiTaskElasticNet
        sklearn.linear_model.BayesianRidge
        sklearn.linear_model.ARDRegression
        sklearn.neural_network.MLPRegressor
        sklearn.neural_network.MLPClassifier
        sklearn.svm.classes.SVC
        sklearn.svm.classes.LinearSVC
        sklearn.svm.classes.NuSVC
        sklearn.svm.classes.SVR
        sklearn.svm.classes.LinearSVR
        sklearn.svm.classes.NuSVR
        sklearn.ensemble.RandomForestClassifier
        sklearn.ensemble.RandomForestRegressor
        sklearn.linear_model.Lasso
    '''
    
    
    
    n_features=X_values.shape[1]
    
    if(type(model_f) is sklearn.linear_model.ElasticNet):
        base_exp=2
        n=2
        ext_z=numpy.array([0]*n) 
        alpha=model_f.get_params().pop('alpha')
        l1_ratio=model_f.get_params().pop('l1_ratio')
        ext_z[0]=log_base(alpha*l1_ratio,base_exp)
        ext_z[1]=log_base(alpha*(1-l1_ratio),base_exp)
        ext_z=ext_z.astype(float)
        ext_lb=numpy.array([-10]*n)
        ext_ub=numpy.array([10]*n)
        ext_step=numpy.array([10**0]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['alpha','l1_ratio']
        has_no_space=[0,0]
        var_is_int=numpy.zeros(n)
        
        
    if(type(model_f) is sklearn.linear_model.MultiTaskElasticNet):        
        base_exp=2
        n=2
        ext_z=numpy.array([0]*n) 
        alpha=model_f.get_params().pop('alpha')
        l1_ratio=model_f.get_params().pop('l1_ratio')
        ext_z[0]=float(log_base(alpha*l1_ratio,base_exp))
        ext_z[1]=float(log_base(alpha*(1-l1_ratio),base_exp))
        ext_z=ext_z.astype(float)
        ext_lb=numpy.array([-10]*n)
        ext_ub=numpy.array([10]*n)
        ext_step=numpy.array([10**0]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['alpha','l1_ratio']
        has_no_space=[0,0]
        var_is_int=numpy.zeros(n)

    if(type(model_f) is sklearn.linear_model.BayesianRidge):
        base_exp=10
        n=4
        ext_z=numpy.array([0]*n)
        alpha_1=log_base(model_f.get_params().pop('alpha_1'),base_exp)
        alpha_2=log_base(model_f.get_params().pop('alpha_2'),base_exp)
        lambda_1=log_base(model_f.get_params().pop('lambda_1'),base_exp)
        lambda_2=log_base(model_f.get_params().pop('lambda_2'),base_exp)
        ext_z=numpy.array([alpha_1,alpha_2,lambda_1,lambda_2]) 
        ext_lb=numpy.array([-20]*n)
        ext_ub=numpy.array([10]*n)
        ext_step=numpy.array([10**0]*n)
        init_int_step=numpy.array([5]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['alpha_1','alpha_2','lambda_1','lambda_2']
        has_no_space=[0,0,0,0]
        var_is_int=numpy.zeros(n)
        
        
    if(type(model_f) is sklearn.linear_model.ARDRegression):
        base_exp=10
        n=4
        ext_z=numpy.array([0]*n)
        alpha_1=log_base(model_f.get_params().pop('alpha_1'),base_exp)
        alpha_2=log_base(model_f.get_params().pop('alpha_2'),base_exp)
        lambda_1=log_base(model_f.get_params().pop('lambda_1'),base_exp)
        lambda_2=log_base(model_f.get_params().pop('lambda_2'),base_exp)
        ext_z=numpy.array([alpha_1,alpha_2,lambda_1,lambda_2]) 
        ext_lb=numpy.array([-20]*n)
        ext_ub=numpy.array([10]*n)
        ext_step=numpy.array([10**0]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['alpha_1','alpha_2','lambda_1','lambda_2']
        has_no_space=[0,0,0,0]
        var_is_int=numpy.zeros(n)

    if(type(model_f) is sklearn.neural_network.MLPRegressor):
        base_exp=10
        n=len(model_f.hidden_layer_sizes)+1
        alpha=log_base(model_f.get_params().pop('alpha'),base_exp)
        hidden_layer_sizes=model_f.get_params().pop('hidden_layer_sizes')
        ext_z=numpy.concatenate((hidden_layer_sizes,[alpha]),axis=0)
        ext_lb=[10]*(n-1)
        ext_lb.append(-10)
        ext_lb=numpy.array(ext_lb)                   
        ext_ub=[1000]*(n-1)
        ext_ub.append(10)
        ext_ub=numpy.array(ext_ub)
        ext_step=[50]*(n-1)
        ext_step.append(1)
        ext_step=numpy.array(ext_step)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['hidden_layer_sizes','alpha']
        has_no_space=[1,0]
        var_is_int=numpy.zeros(n)
        
    if(type(model_f) is sklearn.neural_network.MLPClassifier):
        base_exp=10
        n=len(model_f.hidden_layer_sizes)+1
        alpha=log_base(model_f.get_params().pop('alpha'),base_exp)
        hidden_layer_sizes=model_f.get_params().pop('hidden_layer_sizes')
        ext_z=numpy.concatenate((hidden_layer_sizes,[alpha]),axis=0)
        ext_lb=[10]*(n-1)
        ext_lb.append(-8)
        ext_lb=numpy.array(ext_lb)                   
        ext_ub=[1000]*(n-1)
        ext_ub.append(8)
        ext_ub=numpy.array(ext_ub)
        ext_step=[50]*(n-1)
        ext_step.append(1)
        ext_step=numpy.array(ext_step)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['hidden_layer_sizes','alpha']
        has_no_space=[1,0]
        var_is_int=numpy.zeros(n)


    if(type(model_f) is sklearn.svm.classes.SVC):
        base_exp=2
        n=2
        C=log_base(model_f.get_params().pop('C'),base_exp)
        if model_f.get_params().pop('gamma') is 'auto':
            gamma=log_base(1/n_features,base_exp)
        else:
            gamma=log_base(model_f.get_params().pop('gamma'),base_exp)
        ext_z=numpy.array([C,gamma]) 
        ext_lb=numpy.array([-5,-15])
        ext_ub=numpy.array([15,3])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([5]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['C','gamma']
        has_no_space=[0,0]
        var_is_int=numpy.zeros(n)
        
    if(type(model_f) is sklearn.svm.classes.LinearSVC):
        base_exp=2
        n=1
        ext_z=numpy.array([log_base(model_f.get_params().pop('C'),base_exp)]) 
        ext_lb=numpy.array([-5])
        ext_ub=numpy.array([15])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([5]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['C']
        has_no_space=[0]
        var_is_int=numpy.zeros(n)
        
    if(type(model_f) is sklearn.svm.classes.NuSVC):
        base_exp=2
        n=2
        nu=model_f.get_params().pop('nu')
        if model_f.get_params().pop('gamma') is 'auto':
            gamma=log_base(1/n_features,base_exp)
        else:
            gamma=log_base(model_f.get_params().pop('gamma'),base_exp)
        ext_z=numpy.array([nu,gamma]) 
        ext_lb=numpy.array([0.01,-15])
        ext_ub=numpy.array([0.99,3])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([5]*n)
        ext_is_integer=numpy.array([0,1])
        hp_list=['nu','gamma']
        has_no_space=[1,0]
        var_is_int=numpy.zeros(n)
        
    if(type(model_f) is sklearn.svm.classes.SVR):
        base_exp=2
        n=3
        C=log_base(model_f.get_params().pop('C'),base_exp)
        if model_f.get_params().pop('gamma') is 'auto':
            gamma=log_base(1/n_features,base_exp)
        else:
            gamma=log_base(model_f.get_params().pop('gamma'),base_exp)
        epsilon=log_base(model_f.get_params().pop('epsilon'),base_exp)
        ext_z=numpy.array([C,gamma,epsilon]) 
        ext_lb=numpy.array([-5,-15,-7])
        ext_ub=numpy.array([15,3,3])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['C','gamma','epsilon']
        has_no_space=[0,0,0]
        var_is_int=numpy.zeros(n)
        
    if(type(model_f) is sklearn.svm.classes.LinearSVR):
        base_exp=2
        n=2
        C=log_base(model_f.get_params().pop('C'),base_exp)
        epsilon=log_base(model_f.get_params().pop('epsilon'),base_exp)
        ext_z=numpy.array([C,epsilon]) 
        ext_lb=numpy.array([-5,-7])
        ext_ub=numpy.array([15,3])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['C','epsilon']
        has_no_space=[0,0]
        var_is_int=numpy.zeros(n)
        
    if(type(model_f) is sklearn.svm.classes.NuSVR):
        base_exp=2
        n=3
        C=log_base(model_f.get_params().pop('C'),base_exp)
        nu=model_f.get_params().pop('nu')
        if model_f.get_params().pop('gamma') is 'auto':
            gamma=log_base(1/n_features,base_exp)
        else:
            gamma=log_base(model_f.get_params().pop('gamma'),base_exp)
        ext_z=numpy.array([C,nu,gamma]) 
        ext_lb=numpy.array([-5,0.01,-15])
        ext_ub=numpy.array([15,0.99,3])
        ext_step=numpy.array([1,0.1,1])
        init_int_step=numpy.array([1,0.1,1])
        ext_is_integer=numpy.array([1,0,1])
        hp_list=['C','nu','gamma']
        has_no_space=[0,1,0]
        var_is_int=numpy.zeros(n)
                
    if(type(model_f) is sklearn.ensemble.RandomForestClassifier):
        base_exp=2
        n=5      
        ext_ub=numpy.array([n_features,6,6,15,7])
        max_features=model_f.get_params().pop('max_features')
        if max_features is 'auto':
            max_features=numpy.sqrt(n_features)
        if max_features is 'sqrt':
            max_features=numpy.sqrt(n_features)
        if max_features is 'log2':
            max_features=numpy.log2(n_features)
        
        min_samples_split=log_base(model_f.get_params().pop('min_samples_split'),base_exp)
        min_samples_leaf=log_base(model_f.get_params().pop('min_samples_leaf'),base_exp)
        if model_f.get_params().pop('max_depth') is None:
            max_depth=ext_ub[3]
        else:
            max_depth=log_base(model_f.get_params().pop('max_depth'),base_exp)
        n_estimators=log_base(model_f.get_params().pop('n_estimators'),base_exp)
        ext_z=numpy.array([max_features,min_samples_split,min_samples_leaf,max_depth,n_estimators]) 
        ext_lb=numpy.array([1,1,0,0,3])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([5]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['max_features','min_samples_split','min_samples_leaf','max_depth','n_estimators']
        has_no_space=[1,0,0,0,0]
        var_is_int=numpy.ones(n)
        
    if(type(model_f) is sklearn.ensemble.RandomForestRegressor):
        base_exp=2        
        n=5
        ext_ub=numpy.array([n_features,10,10,15,15])
        max_features=model_f.get_params().pop('max_features')
        if max_features is 'auto':
            max_features=numpy.sqrt(n_features)
        if max_features is 'sqrt':
            max_features=numpy.sqrt(n_features)
        if max_features is 'log2':
            max_features=numpy.log2(n_features)
        min_samples_split=log_base(model_f.get_params().pop('min_samples_split'),base_exp)
        min_samples_leaf=log_base(model_f.get_params().pop('min_samples_leaf'),base_exp)
        if model_f.get_params().pop('max_depth') is None:
            max_depth=ext_ub[3]
        else:
            max_depth=log_base(model_f.get_params().pop('max_depth'),base_exp)
        n_estimators=log_base(model_f.get_params().pop('n_estimators'),base_exp)
        ext_z=numpy.array([max_features,min_samples_split,min_samples_leaf,max_depth,n_estimators]) 
        ext_z=numpy.array([n_features,1,0,0,8]) 
        ext_lb=numpy.array([1,1,0,0,3])
        ext_step=numpy.array([1]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['max_features','min_samples_split','min_samples_leaf','max_depth','n_estimators']
        has_no_space=[1,0,0,0,0]
        var_is_int=numpy.ones(n)


    if(type(model_f) is sklearn.linear_model.Lasso):
        base_exp=2        
        n=1
        alpha=log_base(model_f.get_params().pop('alpha'),base_exp)
        ext_z=numpy.array([alpha]) 
        ext_lb=numpy.array([-100]*n)
        ext_ub=numpy.array([10]*n)
        ext_step=numpy.array([10**0]*n)
        init_int_step=numpy.array([1]*n)
        ext_is_integer=numpy.ones(n)
        hp_list=['alpha']
        has_no_space=[0]
        var_is_int=numpy.zeros(n)
    

    #check if the variable is finite
    for i in range (n):
        if not numpy.isfinite(ext_z[i]) :
            ext_z[i]=ext_lb[i]
    #check if the variables are in bound
    for i in range (n):
        if ext_z[i]>ext_ub[i]:
            raise ValueError('variable %s out of upper bound' % i )
        elif ext_z[i]<ext_lb[i]:
            raise ValueError('variable %s out of lower bound' % i )

            
    
    
    return n,ext_z,ext_lb,ext_ub ,ext_step,init_int_step,ext_is_integer,base_exp,hp_list,has_no_space,var_is_int

def KFold_error():
    '''
    Method that calculates the k-fold error on the training
    '''
    
    global X_values,Y_values,model_f,n_splits, error_measure
    #Get the splits
    kf=KFold(n_splits=n_splits)
    kf.get_n_splits(X_values)
    error_fold=[]

    for train_index, validation_index in kf.split(X_values):
        #Create the splits
        X_train, X_validation = X_values[train_index], X_values[validation_index]
        Y_train, Y_validation = Y_values[train_index], Y_values[validation_index]
        try:
            #Check if the values of the  hyperparameters are alright
            model_f.fit(X_train, Y_train)
        except ValueError as e:
            print("The following error on the values of the hyperparameters has occurred \"%s\", returning infinity" % e)
            return numpy.inf
        F_validation=model_f.predict(X_validation)
        try:
            #Check if the metrics is alright
            error=getattr(sklearn.metrics,error_measure)(Y_validation, F_validation)
        except NameError:
            print("The specified measure of error is not in the module sklearn.metrics, returning the value of the mean squared error")
            error=sklearn.metrics.mean_squared_error(Y_validation, F_validation)
        error_fold.append(error)
    print('error:')
    print(numpy.mean(error_fold))
    return numpy.mean(error_fold)


def set_error(errortype="mse"):
    global X_values,Y_values,model_f,n_splits
    n_training=int(numpy.floor(X_values.shape[0]*0.6))
    X_train=X_values[1:n_training,:]
    Y_train=Y_values[1:n_training]
    X_validation=X_values[n_training:X_values.shape[0],:]
    Y_validation=Y_values[n_training:X_values.shape[0]]
    try:
        model_f.fit(X_train, Y_train)
    except ValueError:
        print("There has been problems with the values of the parameters during training, returning infinity")
        return numpy.inf
    F_validation=model_f.predict(X_validation)
    try:
        error=getattr(sklearn.metrics,error_measure)(Y_validation, F_validation)
    except NameError:
        print("The specified error is not in the module sklearn.metrics, returning the value of the mean squared error")
        error=sklearn.metrics.mean_squared_error(Y_validation, F_validation)
    return error




def training_set_standardization(X_training):
    # normalization / standardization
    mean_ = X_training.sum(axis=0) / X_training.shape[0]
    # Q = ((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0]
    # std_ = numpy.array([numpy.sqrt(Q[i]) for i in range(Q.shape[0])])

    std_ = numpy.sqrt((((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0]))
    # Xn_training = scale(X_training)

    Xn_training = (X_training - mean_)/ std_

    return Xn_training, mean_, std_


def training_set_destandardization(X_training, mean_, std_):
    return X_training * std_ + mean_

def interface_funct(model_f,z):
    '''
    function that changes the values of the hyperparameters in the estimator
    
    Note that because of the peculiarity of the multilayer neural networks, a 
    custom made method has been created. 
    This method can be a template for the creation of custom functions to 
    change the values of the hyperparameters.
    '''
    
    
    global hp_list,has_no_space,base_exp,var_is_int,space_points,hp_space
    dict_={}

    y=[0]*len(z)
    if(type(model_f) is sklearn.neural_network.MLPRegressor or type(model_f)  is sklearn.neural_network.MLPClassifier):
        dict_['hidden_layer_sizes']=tuple(z[0:len(z)-1].astype(int))
        y[0]=tuple(z[0:len(z)-1].astype(int))
        if has_no_space[len(z)-1]:
            dict_['alpha']=z[len(z)-1]
            y[1]=z[len(z)-1]
        else:
            dict_['alpha']=space_points[len(z)-1][int(z[len(z)-1])]
            y[1]=space_points[len(z)-1][int(z[len(z)-1])]
    else:    
        for i in range(len(z)):
            if has_no_space[i]:
                if var_is_int[i]:
                   dict_[hp_list[i]]=int(z[i]) 
                else:
                    dict_[hp_list[i]]=z[i]
            else:
                if var_is_int[i]:
                   dict_[hp_list[i]]=int(space_points[i][int(z[i])])
                   y[i]=int(space_points[i][int(z[i])])
                else:
                    dict_[hp_list[i]]=space_points[i][int(z[i])]
                    y[i]=space_points[i][int(z[i])]

    print(y)

    model_f.set_params(**dict_)


def model_funct(z,n):
    '''
    function that is called directly by dfl in Fortran
    '''
    global model_f, base_exp
    
    print(z)
    interface_funct(model_f,z)
    return KFold_error()


class DFLScikit(BaseEstimator):

    def __init__(self,estimator,custom=0,n=None,loss='mean_squared_error',hp_list=None,has_no_space=None,var_is_int=None,hp_space=['default'],space_num=None,z0=None,lb=None,ub=None,is_integer=None,step=None,init_int_step=None,alfa_stop=10**-3, nf_max=200, iprint=2, kf_splits=5,base=None):
        '''
        The interface to apply the DFL algorithm to the methods implemented in 
        scikit learn.
        DFL is a derivative free algorithm applied to the problem of optimizing
        the hyperparameters of a machine learning method. 
        The choice of a derivative free optimization method is due to the 
        impossibility to obtain the relation between the hyperparameters and 
        the error function in closed form.
        The algorithm is capable of handling both integer and continuous 
        variables and the sequences it produces are characterized by the 
        convergence to stationary points of the objective function.

        DFL is available on the site: http://www.diag.uniroma1.it/~lucidi/DFL/        
        
        Parameters
        ----------          
        estimator : scikit-learn estimator, 
            The model which hyperparameters must be optimized.
            
        custom : boolean, optional (default=0)
            It indicates if the estimator must be handled by the 
            start_up_interface method not. 
            In the case its value is one, a custom made estimator can be used, 
            or other methods present in scikit-learn but not comprised in 
            start_up_interface method can be applied.
            
            
        n : interger, optional (default=None)
            Number of hyperparamters to optimize
            
        loss : str, optional (default='mean_squared_error')
            The loss function used as the derivative free objective function. 
            The algorithm uses the measures of error available in the module 
            sklearn.metrics, therefore the value of this string must be 
            the name of one of the methods available in such module.
        
        hp_list : dict, optional (default= None)
            Dict of str containing the names of the hyperparameters. The names 
            the user wants to include must be the same as the parameters of 
            the estimator.
        
        has_no_space : boolean optional (default= None)
            If 0 it indicates if the variable moves on the place indicated by 
            hp_space, if 1 the variable moves in the set of real number if it 
            is continuous, and in the set of integer numbers if it is interger 
            as indicated in the parameter "is_integer".
        
        var_is_int : boolean optional (default= None)
            it indicates if the estimator chosen to be optimizzed requires the 
            variable must be of type int.
            
            Note that certain methods like RandomForestRegressor or 
            RandomForestClassifier require that the values of the 
            hyperparameters must be int, while the code in Fortran 
            automatically returns to python float values, therefore this 
            parameter is required to not general ValueErrors in the execution 
            of the code.
            
        hp_space : list optional (default='default')
            It indicates what kind of values the meshes of the grid assumes for 
            the ith variable. 
            
            Note that if has_no_space is 1 for the variable, this parameter is 
            ignored.
            If 'default' it generates the following meshes for the 
            hyperparameter: base^(lb),base^(lb+1),...,base^(ub-1),base^(ub).
            if 'linear' it generates the meshes on a linear scale according to
            the method numpy.linspace(lb[i],ub[i],num=space_num[i]).
            if 'log' it generates the the meshes on a log scale according to
            the method numpy.logspace(ext_lb[i],ext_ub[i],
                                      num=self.space_num[i],base=base)
            
        space_num : list optional (default= None)
            List of integer containing the number of points on the meshes for 
            every hyperparameter if 'linear' and 'log' are used in hp_space
            
        z0 : list optional (default= None)
            Initial vector for the values of the hyperparameters
            
        lb : list optional (default= None)
            Lower bound on the values of the hyper parameters
            
        ub :list optional (default= None)
            Upper bound on the values of the hyper parameters
            
        is_integer : list optional (default= None)
            List of booleans. If 1 it indicates that the associate parameters 
            must assume discrete values, if 0 it indicate that the associate 
            hyperparameter can assume continuous values. 
            
            Note that In order to accelerate the optimization,the 
            hyperparameters are generally moved on a suitable grid even if they 
            are continuous. 
            If the user wants the i-th hyperparameter to be moved on the space 
            of the real numbers he must set:
                has_no_space[i]=1
                is_integer[i]=0
            If the user wants the i-th hyperparameter to be moved on the space 
            of the integer numbers he must set:   
                has_no_space[i]=1
                is_integer[i]=1
            If the user wants the i-th hyperparameter to be moved on the space 
            Defined by the list hp_space he must set:   
                has_no_space[i]=0
                is_integer[i]=1
            
        step : list optional (default= None)
            List of minimum steps to be taken along the integer hyperparameters
            
        init_int_step : list optional (default= None)
            List of initial steps
            
        alfa_stop : float optional (default= 10**-3)
            Minimum step size along the continuous hyperparameters
            
        nf_max : integer optional (default= 200)
            Maximum number of fuction evaluations for DFL
            
        iprint : integer optional (default= 2)
            Level of printing for DFL
            
        kf_splits : integer optional (default= 5)
            Number of splits in the k-fold cross validation
            
        base : float optional (default= 2)
            Base of the exponential mesh of the grid
        '''
        self.estimator=estimator
        self.custom=custom
        self.loss=loss
        self.hp_list=hp_list
        self.has_no_space=has_no_space
        self.var_is_int=var_is_int
        self.hp_space=hp_space
        self.space_num=space_num
        self.n=n
        self.z0=z0
        self.lb=lb
        self.ub=ub
        self.is_integer=is_integer
        self.step=step
        self.init_int_step=init_int_step
        self.alfa_stop =alfa_stop
        self.nf_max=nf_max
        self.iprint=iprint
        self.kf_splits=kf_splits
        self.base=base
    
    
    def fit(self,X,Y):
        '''
        Standard fit method for a scikit-learn estimator
        '''        
        global model_f,base_exp,X_values,Y_values,n_splits, error_measure,hp_list,has_no_space,var_is_int
                
        X_values=X
        Y_values=Y
        n_splits=self.kf_splits
        model_f=self.estimator
        n=None
        error_measure=self.loss
        
        
        # Checks if the estimator provided by the user is considered custom, if it not the parameters necessary for the optimization are set in the start_up_interface method
        if not self.custom:
            n,ext_z,ext_lb,ext_ub ,ext_step,init_int_step,ext_is_integer,base_exp,hp_list,has_no_space,var_is_int=start_up_interface(model_f,self.hp_list,self.has_no_space,self.var_is_int)
    
            
        if n is None:
            print("The estimator is not natively supported by DFLsci, please be sure to have the estimator interface for the model_funct method, or be sure that the value of the custom parameter is 1")
            n=self.n
            if not isinstance(n,int):
                raise ValueError('The value of n has not been set to an integer value, impossible to proceed, check if DFLsci is compatible with the current estimator or set n to an integer value ')
        


        #sets the values of the parameters to those provided by the user, otherwise stores the default values provided by start_up_interface in the parameters of the DFLScikit object


        if self.hp_list is None:
            self.hp_list=hp_list
        else:
            hp_list=self.hp_list

        if self.has_no_space is None:
            self.has_no_space=has_no_space
        else:
            has_no_space=self.has_no_space
            
        if (self.var_is_int) is None:
            self.var_is_int=var_is_int
        else:
            var_is_int=self.var_is_int

        if self.hp_space == ['default']:
            self.hp_space=['default']*n

        if self.space_num is None:
            self.space_num=[1]*n

        if self.z0 is  None:
            self.z0=ext_z
        else:
            ext_z=self.z0 

        if self.lb is  None:    
            self.lb=ext_lb
        else:
            ext_lb=self.lb

        if self.ub is None:    
            self.ub=ext_ub
        else:
            ext_ub=self.ub

        if self.is_integer is None:
            self.is_integer=ext_is_integer
        else:
            ext_is_integer=self.is_integer
            
        if self.step is None:
            self.step=ext_step
            if self.hp_space is not 'default':
                ext_step=numpy.array([1]*n)
                self.step=ext_step
        else:
            ext_step=self.step
                          
        if self.init_int_step is  None:
            self.init_int_step=init_int_step
            if self.hp_space is not 'default':
                init_int_step=numpy.array([1]*n)
                self.init_int_step=init_int_step
        else:
            init_int_step=self.init_int_step

        if self.base is  None:
            self.base=base_exp
        else:
            base_exp=self.base

        #Creates the meshes of the grid where the hyperparameters must be moved
        ext_z,ext_lb,ext_ub,ext_is_integer=self.create_meshes(n,ext_z,ext_lb,ext_ub,ext_is_integer)
         
        #setting the common parameters in Fortran
        dfl.ext.ext_alfa_stop=self.alfa_stop
        dfl.ext.ext_nf_max=self.nf_max
        dfl.ext.ext_iprint=self.iprint


        #Calls the optimization method
        result_z=dfl.main_box_discr(model_funct,ext_z,ext_lb,ext_ub,ext_step,init_int_step,ext_is_integer,n)

        #Final training with all optimal hyperparameters and all the samples in the training set
        self.estimator.fit(X_values,Y_values)
    
        return self

    def predict(self,X):
        return self.estimator.predict(X)

    def create_meshes(self,n,ext_z,ext_lb,ext_ub,ext_is_integer):
        '''
        method that creates the meshes where the hyperparameters must be moved
        
        Note if has_no_space is 1 the chosen variables does not move in the 
        meshes but rather in the space or real numbers if it is continuous or 
        in the space of integer numbers if it is discrete all according to the 
        list is_integer.
        '''
         
        global space_points
        space_points=[1]*n    
        for i in range(n):
            if self.has_no_space[i]:
                space_points[i]='continous'
            else:
                if self.hp_space[i] is 'default':
                    self.space_num[i]=int(ext_ub[i]-ext_lb[i]+1)
                    space_points[i]=numpy.logspace(ext_lb[i],ext_ub[i],self.space_num[i],base=base_exp)
                    ext_z[i]=base_exp**ext_z[i]
                elif self.hp_space[i] is 'linear':
                    space_points[i]=numpy.linspace(ext_lb[i],ext_ub[i],num=self.space_num[i])
                elif self.hp_space[i] is 'log':
                    space_points[i]=numpy.logspace(ext_lb[i],ext_ub[i],num=self.space_num[i],base=base_exp)
                ext_lb[i]=0
                ext_ub[i]=self.space_num[i]-1
                ext_is_integer[i]=1
                if ext_z[i] not in space_points[i]:
                    temp=[0]*self.space_num[i]
                    for j in range(self.space_num[i]):
                        temp[j]=(numpy.absolute(ext_z[i]-space_points[i][j]))
                    index_=temp.index(min(temp))    
                    ext_z[i]=index_#space_points[i][index_]
                    print('variable %s not on the grid, starting from the closest point instead' % i)
                else:
                    ext_z[i]=space_points[i].tolist().index(ext_z[i])
                
        return ext_z,ext_lb,ext_ub,ext_is_integer
    

        
if __name__ == '__main__':
    dataset=sklearn.datasets.load_boston()
    #dataset=sklearn.datasets.load_breast_cancer()
    X=dataset.data
    Y=dataset.target
    X, mean_, std_=training_set_standardization(dataset.data)
    Y, meanY_, stdY_=training_set_standardization(dataset.target)
    error=[]
    #for i in range(1,10):
    #network=sklearn.linear_model.ElasticNet()
    #network=sklearn.linear_model.MultiTaskElasticNet()
    #network=sklearn.linear_model.BayesianRidge()
    #network=sklearn.linear_model.ARDRegression()
    #network=sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(10, ),random_state=0)
    network=sklearn.svm.SVR()
    #network=sklearn.svm.LinearSVR()
    #network=sklearn.svm.NuSVR()
    #network=sklearn.ensemble.RandomForestRegressor(random_state=1)
    #network=sklearn.linear_model.Lasso()
    #network=sklearn.svm.SVC()
    #network=sklearn.svm.LinearSVC()
    #network=sklearn.svm.NuSVC()
    #network=sklearn.neural_network.MLPClassifier()
    #network=sklearn.ensemble.RandomForestClassifier()
    n=3
    #model=DFLScikit(network,hp_space=['log']*n,space_num=[50]*n)
    #model=DFLScikit(network,hp_space=['linear']*n,space_num=[100]*n,lb=[5,5,5,5,5],ub=[10,10,10,10,10])
    #model=DFLScikit(network,var_is_int=[0,0,0])
    model=DFLScikit(network,custom=1,base=2,n=n,hp_list=['C','gamma','epsilon'], z0=numpy.array([2,2,0.5]), lb=numpy.array([-5,-15,-7]), ub=numpy.array([15,3,3]), step=numpy.array([1]*n), init_int_step=numpy.array([1]*n),is_integer=numpy.ones(n),has_no_space=[0,0,0],var_is_int=numpy.zeros(n))

    
    
    model.fit(X,Y)
    
    #lb=[0,0.01,0],ub=[1000,0.99,8]

