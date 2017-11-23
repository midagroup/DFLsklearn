#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:19:22 2017

@authors: V. Latorre, F. Benvenuto
"""
# f2py -c dfl.pyf sd.f90 main_box.f90
import numpy
import sklearn.linear_model
import dfl
import sys
from sklearn.model_selection import KFold
import sklearn.datasets
import sklearn.neural_network
import sklearn.svm
import sklearn.ensemble
from sklearn.base import BaseEstimator


def log_base(x, base):
    """
    A simple function to calculate the logarithm of x in the base contained in
    the argument 'base'

    """
    return numpy.log(x) / numpy.log(base)


def configuration_preset(model_f, X_values):
    """
    Method that automatically initializes some of the estimators of
    Scikit-learn for a more user-friendly interface. The currently supported
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
    """

    n_features = X_values.shape[1]

    if(isinstance(model_f, sklearn.linear_model.ElasticNet)):
        base_exp = 2
        n = 2
        ext_z = numpy.array([0] * n)
        alpha = model_f.get_params().pop('alpha')
        l1_ratio = model_f.get_params().pop('l1_ratio')
        ext_z = numpy.array([log_base(alpha, base_exp), l1_ratio])
        ext_z = ext_z.astype(float)
        ext_lb = numpy.array(-10, 0.0)
        ext_ub = numpy.array(10, 1.0)
        ext_step = numpy.array([10**0] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['alpha', 'l1_ratio']
        on_a_mesh = [1, 0]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.linear_model.MultiTaskElasticNet)):
        base_exp = 2
        n = 2
        ext_z = numpy.array([0] * n)
        alpha = model_f.get_params().pop('alpha')
        l1_ratio = model_f.get_params().pop('l1_ratio')
        ext_z = numpy.array([log_base(alpha, base_exp), l1_ratio])
        ext_z = ext_z.astype(float)
        ext_lb = numpy.array(-10, 0.0)
        ext_ub = numpy.array(10, 1.0)
        ext_step = numpy.array([10**0] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['alpha', 'l1_ratio']
        on_a_mesh = [1, 0]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.linear_model.BayesianRidge)):
        base_exp = 10
        n = 4
        ext_z = numpy.array([0] * n)
        alpha_1 = log_base(
            model_f.get_params().pop('alpha_1'),
            base_exp)
        alpha_2 = log_base(
            model_f.get_params().pop('alpha_2'),
            base_exp)
        lambda_1 = log_base(
            model_f.get_params().pop('lambda_1'), base_exp)
        lambda_2 = log_base(
            model_f.get_params().pop('lambda_2'), base_exp)
        ext_z = numpy.array([alpha_1, alpha_2, lambda_1, lambda_2])
        ext_lb = numpy.array([-20] * n)
        ext_ub = numpy.array([10] * n)
        ext_step = numpy.array([10**0] * n)
        init_int_step = numpy.array([5] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']
        on_a_mesh = [1, 1, 1, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.linear_model.ARDRegression)):
        base_exp = 10
        n = 4
        ext_z = numpy.array([0] * n)
        alpha_1 = log_base(
            model_f.get_params().pop('alpha_1'),
            base_exp)
        alpha_2 = log_base(
            model_f.get_params().pop('alpha_2'),
            base_exp)
        lambda_1 = log_base(
            model_f.get_params().pop('lambda_1'), base_exp)
        lambda_2 = log_base(
            model_f.get_params().pop('lambda_2'), base_exp)
        ext_z = numpy.array([alpha_1, alpha_2, lambda_1, lambda_2])
        ext_lb = numpy.array([-20] * n)
        ext_ub = numpy.array([10] * n)
        ext_step = numpy.array([10**0] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']
        on_a_mesh = [1, 1, 1, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.neural_network.MLPRegressor)):
        base_exp = 10
        n = len(model_f.hidden_layer_sizes) + 1
        alpha = log_base(model_f.get_params().pop('alpha'), base_exp)
        hidden_layer_sizes = model_f.get_params().pop('hidden_layer_sizes')
        ext_z = numpy.concatenate((hidden_layer_sizes, [alpha]), axis=0)
        ext_lb = [10] * (n - 1)
        ext_lb.append(-10)
        ext_lb = numpy.array(ext_lb)
        ext_ub = [1000] * (n - 1)
        ext_ub.append(8)
        ext_ub = numpy.array(ext_ub)
        ext_step = [50] * (n - 1)
        ext_step.append(1)
        ext_step = numpy.array(ext_step)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['hidden_layer_sizes', 'alpha']
        on_a_mesh = [0, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.neural_network.MLPClassifier)):
        base_exp = 10
        n = len(model_f.hidden_layer_sizes) + 1
        alpha = log_base(model_f.get_params().pop('alpha'), base_exp)
        hidden_layer_sizes = model_f.get_params().pop('hidden_layer_sizes')
        ext_z = numpy.concatenate((hidden_layer_sizes, [alpha]), axis=0)
        ext_lb = [10] * (n - 1)
        ext_lb.append(-10)
        ext_lb = numpy.array(ext_lb)
        ext_ub = [1000] * (n - 1)
        ext_ub.append(8)
        ext_ub = numpy.array(ext_ub)
        ext_step = [50] * (n - 1)
        ext_step.append(1)
        ext_step = numpy.array(ext_step)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['hidden_layer_sizes', 'alpha']
        on_a_mesh = [0, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.svm.classes.SVC)):
        base_exp = 2
        n = 2
        C = log_base(model_f.get_params().pop('C'), base_exp)
        if model_f.get_params().pop('gamma') is 'auto':
            gamma = log_base(1 / n_features, base_exp)
        else:
            gamma = log_base(
                model_f.get_params().pop('gamma'), base_exp)
        ext_z = numpy.array([C, gamma])
        ext_lb = numpy.array([-15, -5])
        ext_ub = numpy.array([3, 15])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([5] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['gamma', 'C']
        on_a_mesh = [1, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.svm.classes.LinearSVC)):
        base_exp = 2
        n = 1
        ext_z = numpy.array(
            [log_base(model_f.get_params().pop('C'), base_exp)])
        ext_lb = numpy.array([-5])
        ext_ub = numpy.array([15])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([5] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['C']
        on_a_mesh = [1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.svm.classes.NuSVC)):
        base_exp = 2
        n = 2
        nu = model_f.get_params().pop('nu')
        if model_f.get_params().pop('gamma') is 'auto':
            gamma = log_base(1 / n_features, base_exp)
        else:
            gamma = log_base(
                model_f.get_params().pop('gamma'), base_exp)
        ext_z = numpy.array([nu, gamma])
        ext_lb = numpy.array([0.01, -15])
        ext_ub = numpy.array([0.99, 3])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([5] * n)
        ext_is_integer = numpy.array([0, 1])
        hp_list = ['nu', 'gamma']
        on_a_mesh = [0, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.svm.classes.SVR)):
        base_exp = 2
        n = 3
        C = log_base(model_f.get_params().pop('C'), base_exp)
        if model_f.get_params().pop('gamma') is 'auto':
            gamma = log_base(1 / n_features, base_exp)
        else:
            gamma = log_base(
                model_f.get_params().pop('gamma'), base_exp)
        epsilon = log_base(
            model_f.get_params().pop('epsilon'),
            base_exp)
        ext_z = numpy.array([gamma, C, epsilon])
        ext_lb = numpy.array([-15, -5, -7])
        ext_ub = numpy.array([3, 15, 3])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['gamma', 'C', 'epsilon']
        on_a_mesh = [1, 1, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.svm.classes.LinearSVR)):
        base_exp = 2
        n = 2
        C = log_base(model_f.get_params().pop('C'), base_exp)
        epsilon = log_base(
            model_f.get_params().pop('epsilon'),
            base_exp)
        ext_z = numpy.array([C, epsilon])
        ext_lb = numpy.array([-5, -7])
        ext_ub = numpy.array([15, 3])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['C', 'epsilon']
        on_a_mesh = [1, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.svm.classes.NuSVR)):
        base_exp = 2
        n = 3
        C = log_base(model_f.get_params().pop('C'), base_exp)
        nu = model_f.get_params().pop('nu')
        if model_f.get_params().pop('gamma') is 'auto':
            gamma = log_base(1 / n_features, base_exp)
        else:
            gamma = log_base(
                model_f.get_params().pop('gamma'), base_exp)
        ext_z = numpy.array([C, nu, gamma])
        ext_lb = numpy.array([-5, 0.01, -15])
        ext_ub = numpy.array([15, 0.99, 3])
        ext_step = numpy.array([1, 0.1, 1])
        init_int_step = numpy.array([1, 0.1, 1])
        ext_is_integer = numpy.array([1, 0, 1])
        hp_list = ['C', 'nu', 'gamma']
        on_a_mesh = [1, 0, 1]
        var_is_int = numpy.zeros(n)

    if(isinstance(model_f, sklearn.ensemble.RandomForestClassifier)):
        base_exp = 2
        n = 5
        ext_ub = numpy.array([n_features, 6, 6, 15, 7])
        max_features = model_f.get_params().pop('max_features')
        if max_features is 'auto':
            max_features = numpy.sqrt(n_features)
        if max_features is 'sqrt':
            max_features = numpy.sqrt(n_features)
        if max_features is 'log2':
            max_features = numpy.log2(n_features)

        min_samples_split = log_base(
            model_f.get_params().pop('min_samples_split'), base_exp)
        min_samples_leaf = log_base(
            model_f.get_params().pop('min_samples_leaf'), base_exp)
        if model_f.get_params().pop('max_depth') is None:
            max_depth = ext_ub[3]
        else:
            max_depth = log_base(
                model_f.get_params().pop('max_depth'), base_exp)
        n_estimators = log_base(
            model_f.get_params().pop('n_estimators'), base_exp)
        ext_z = numpy.array([max_features,
                             min_samples_split,
                             min_samples_leaf,
                             max_depth,
                             n_estimators])
        ext_lb = numpy.array([1, 1, 0, 0, 3])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([5] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = [
            'max_features',
            'min_samples_split',
            'min_samples_leaf',
            'max_depth',
            'n_estimators']
        on_a_mesh = [0, 1, 1, 1, 1]
        var_is_int = numpy.ones(n)

    if(isinstance(model_f, sklearn.ensemble.RandomForestRegressor)):
        base_exp = 2
        n = 5
        ext_ub = numpy.array([n_features, 10, 10, 15, 15])
        max_features = model_f.get_params().pop('max_features')
        if max_features is 'auto':
            max_features = numpy.sqrt(n_features)
        if max_features is 'sqrt':
            max_features = numpy.sqrt(n_features)
        if max_features is 'log2':
            max_features = numpy.log2(n_features)
        min_samples_split = log_base(
            model_f.get_params().pop('min_samples_split'), base_exp)
        min_samples_leaf = log_base(
            model_f.get_params().pop('min_samples_leaf'), base_exp)
        if model_f.get_params().pop('max_depth') is None:
            max_depth = ext_ub[3]
        else:
            max_depth = log_base(
                model_f.get_params().pop('max_depth'), base_exp)
        n_estimators = log_base(
            model_f.get_params().pop('n_estimators'), base_exp)
        ext_z = numpy.array([max_features,
                             min_samples_split,
                             min_samples_leaf,
                             max_depth,
                             n_estimators])
        ext_z = numpy.array([n_features, 1, 0, 0, 8])
        ext_lb = numpy.array([1, 1, 0, 0, 3])
        ext_step = numpy.array([1] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = [
            'max_features',
            'min_samples_split',
            'min_samples_leaf',
            'max_depth',
            'n_estimators']
        on_a_mesh = [0, 1, 1, 1, 1]
        var_is_int = numpy.ones(n)

    if(isinstance(model_f, sklearn.linear_model.Lasso)):
        base_exp = 2
        n = 1
        alpha = log_base(model_f.get_params().pop('alpha'), base_exp)
        ext_z = numpy.array([alpha])
        ext_lb = numpy.array([-100] * n)
        ext_ub = numpy.array([10] * n)
        ext_step = numpy.array([10**0] * n)
        init_int_step = numpy.array([1] * n)
        ext_is_integer = numpy.ones(n)
        hp_list = ['alpha']
        on_a_mesh = [1]
        var_is_int = numpy.zeros(n)

    return n, ext_z, ext_lb, ext_ub, ext_step, init_int_step, ext_is_integer, base_exp, hp_list, on_a_mesh, var_is_int


class DFL_estimator(BaseEstimator):
    """
    The interface to apply the DFL algorithm to the methods implemented in
    Scikit-learn for Hyperparameters optimization.
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
    estimator_path : str, optional, default None
        extended path of the estimator, for example:
        'sklearn.neural_network.MLPRegressor'

    estimator_param : dict, optional, default {}
        Parameters to initialize the estimator

    preset_config : boolean, optional, default 1
        It indicates if the estimator must be handled by the
        configuration_preset method or not.
        In the case its value is one, the values of the formal arguments
        that has not been provided by the user will be set to the default
        values of configuration_preset

    preset_function : pointer to a function, optional, default None
        it indicates the name of the method one wants to use to initialize
        the optimization parameters instead of configuration_preset.

    hp_list : dict, optional, default None
        Dict of str containing the names of the hyperparameters. The names
        the user wants to include must be among the estimator 
	hyperparameters obtained by the .get_params() function of Sklearn.

    hp_init : list optional, default None
        Initial vector of the values of the hyperparameters

    on_a_mesh : boolean, optional, default None
        If 1 it indicates if the variable moves on the space indicated by
        hp_mesh_bin, if 0 the variable moves in the set of real number if
        it is continuous, and in the set of integer numbers if it is
        interger as indicated in the parameter "is_integer".

    hp_mesh_bin : list, optional, default ['log']
        It indicates what kind of values the meshes of the grid assumes for
        the ith variable.
        Note that if on_a_mesh is 0 for the variable, this parameter is
        ignored.
        if 'log' it generates the the meshes on a log scale according to
        the method numpy.logspace(lb[i],ub[i],num=self.bin_num[i],base=base).
        if 'linear' it generates the meshes on a linear scale according to
        the method numpy.linspace(lb[i],ub[i],num=bin_num[i]).

        if bin_num is None for that variable then:
            if 'log' it generates the the meshes on a log scale according to
            the method numpy.logspace(lb[i],ub[i],num=ub[i]-lb[i],base=base).
            if 'linear' the variable is handled like an integer variable

    bin_num : list, optional, default None
        List of integer containing the number of bins on the meshes for
        every hyperparameter

    base : float, optional, default 2
        Base of the exponential mesh of the grid

    lb : list, optional, default None
        List of lower bounds values of the hyperparameters

        Note that if on_a_mesh[i] is 0, lb[i] is the lower value that
        the ith variable can assume, while if on_a_mesh[i] is 1 then
        lb[i] indicates the lowest value that the exponent of the lower
        bound on the hyperparameter can assume  based on the  base of the
        exponential.
        in other words: z[i]>base^lb[i]

    ub : list, optional, default None
        List of upper bounds values of the hyper parameters

        Note that if on_a_mesh[i] is 0, ub[i] is the upper value that
        the ith variable can assume, while if on_a_mesh[i] is 1 then
        ub[i] indicates the highest value that the exponent of the upper
        bound on the hyperparameter can assume  based on the  base of the
        exponential.
        in other words: z[i]<base^ub[i]

    metric : str, optional, default 'mean_squared_error'
        The metric function used as the derivative free objective function.
        The algorithm uses the measures of error available in the module
        sklearn.metrics, therefore the value of this string must be
        the name of one of the methods available in such module.

    kf_splits : integer, optional, default 5
        Number of splits in the k-fold cross validation

    minimization : boolean, optional, default True
        It indicates if the cross validation function should be minimizzed
        or maximized

    is_integer : list, optional, default None
        List of booleans. If 1 it indicates that the associate parameters
        must assume discrete values, if 0 it indicate that the associate
        hyperparameter can assume continuous values.

        Note that in order to accelerate the optimization,the
        hyperparameters can be moved on a suitable grid even if they
        are continuous.
        If the user wants the i-th hyperparameter to be moved on the space
        of the real numbers he must set:
        on_a_mesh[i]=0
        is_integer[i]=0
        If the user wants the i-th hyperparameter to be moved on the space
        of the integer numbers he must set:
        on_a_mesh[i]=0
        is_integer[i]=1
        If the user wants the i-th hyperparameter to be moved on the space
        Defined by the list hp_mesh_bin he must set:
        on_a_mesh[i]=1
	then is_integer[i] will be overidden to 1 

    step : list, optional, default None
        List of minimum steps to be taken along the integer hyperparameters

    init_int_step : list, optional, default None
        List of initial steps

    alfa_stop : float, optional, default 10**-3
        Minimum step size along the continuous hyperparameters

    nf_max : integer, optional, default 200
        Maximum number of fuction evaluations for DFL

    iprint : integer, optional, default 2
        Level of printing for DFL

    var_is_int : boolean, optional, default None
        it indicates if the estimator chosen to be optimizzed requires the
        variable to be of type int.

        Note that certain methods like RandomForestRegressor or
        RandomForestClassifier require that the values of the
        hyperparameters must be of type int, while the code in Fortran
        automatically returns to python float values, therefore this
        parameter is required to not have ValueErrors in the execution
        of the code.
    
    """

    def __init__(
            self,
            estimator_path=None,
            estimator_param={},
            preset_config=1,
            preset_function=None,
            hp_list=None,
            hp_init=None,
            on_a_mesh=None,
            hp_mesh_bin=['log'],
            bin_num=None,
            base=None,
            lb=None,
            ub=None,
            metric='mean_squared_error',
            kf_splits=5,
            minimization=True,
            is_integer=None,
            step=None,
            init_int_step=None,
            alfa_stop=10**-3,
            nf_max=200,
            iprint=2,
            var_is_int=None):

        self.estimator_path = estimator_path
        self.estimator_param = estimator_param
        self.preset_config = preset_config
        self.preset_function = preset_function
        self.metric = metric
        self.minimization = minimization
        self.hp_list = hp_list
        self.on_a_mesh = on_a_mesh
        self.var_is_int = var_is_int
        self.hp_mesh_bin = hp_mesh_bin
        self.bin_num = bin_num
        self.hp_init = hp_init
        self.lb = lb
        self.ub = ub
        self.is_integer = is_integer
        self.step = step
        self.init_int_step = init_int_step
        self.alfa_stop = alfa_stop
        self.nf_max = nf_max
        self.iprint = iprint
        self.kf_splits = kf_splits
        self.base = base

        self.n = None
        if self.hp_list is not None:
            if self.hp_init is not None:
                for i in range(len(self.hp_init)):
                    self.estimator_param[self.hp_list[i]] = self.hp_init[i]
                aux_n,aux_hp_init=self.start_up_hp_init()
                self.n = aux_n
                self.hp_init = aux_hp_init

        separator_index = estimator_path[::-1].find('.')
        lib_path = estimator_path[0:-separator_index - 1]
        estimator_path_aux = estimator_path[-separator_index:len(
            estimator_path)]
        self.estimator = getattr(
            sys.modules[lib_path],
            estimator_path_aux)(
            **self.estimator_param)

    def start_up_hp_init(self):
        """
        Method that sets the number of hyperparameters to be considered by DFL
        
        Note that if a hyperparamter belongs to a more complex data structure 
        like a list, a dictionary or a tuple in Python, every single component 
        of such structure must be considered a variable in DFL. Therefore this 
        method checks if the hyperparameter belongs to one of such data 
        structures and set n, that should be considered by DFL, accordingly.
        The same thing is done for the vector with the initial values of the
        hyperparameters
        """
        aux_n = 0
        aux_hp_init = []

        for i in range(len(self.hp_init)):
            if hasattr(self.hp_init[i], '__len__'):
                aux_n = aux_n + len(self.hp_init[i])
                for j in range(len(self.hp_init[i])):
                    aux_hp_init.append(self.hp_init[i][j])
            else:
                aux_n = aux_n + 1
                aux_hp_init.append(self.hp_init[i])
        return aux_n, aux_hp_init
        

    def fit(self, X, Y):
        """
        Standard fit method for a Scikit-learn estimator
        """

        self.X_values = X
        self.Y_values = Y
        n = None

        # Checks if the estimator provided by the user is considered custom, if
        # it not the parameters necessary for the optimization are set in the
        # configuration_preset method
        if self.preset_config:
            if self.preset_function is None:
                n, ext_z, ext_lb, ext_ub, ext_step, init_int_step, ext_is_integer, base_exp, hp_list, on_a_mesh, var_is_int = configuration_preset(
                    self.estimator, self.X_values)
            else:
                n, ext_z, ext_lb, ext_ub, ext_step, init_int_step, ext_is_integer, base_exp, hp_list, on_a_mesh, var_is_int = self.preset_function(
                    self.estimator)
            aux_hp_init = []
            for i in range(len(ext_z)):
                if hasattr(ext_z[i], '__len__'):
                    for j in range(len(ext_z[i])):
                        aux_hp_init.append(ext_z[i][j])
                else:
                    aux_hp_init.append(ext_z[i])

            ext_z = aux_hp_init

        else:
            print(
                'Custom mode active. Please check if all the necessary '
                'formal parameters have been initialized as written at the '
                'beginning of the "configuration_preset" method otherwise '
                'errors could occur')

        if self.n is None:
            self.n = n
        else:
            n = self.n

        if not isinstance(self.n, int):
            raise ValueError(
                'The value of n has not been set to an integer '
                'value, impossible to proceed, check if '
                'DFLsklearn is compatible with the current '
                'estimator or check the setting of the formal '
                'arguments hp_list and hp_init ')

        # sets the values of the parameters to those provided by the user,
        # otherwise stores the default values provided by configuration_preset
        # in the parameters of the DFLscikit object

        if self.hp_list is None:
            self.hp_list = hp_list
        else:
            hp_list = self.hp_list

        if self.on_a_mesh is None:
            self.on_a_mesh = on_a_mesh
        else:
            on_a_mesh = self.on_a_mesh

        if (self.var_is_int) is None:
            self.var_is_int = var_is_int
        else:
            var_is_int = self.var_is_int

        if self.hp_mesh_bin == ['default']:
            self.hp_mesh_bin = ['default'] * self.n

        if self.hp_init is None:
            self.hp_init = ext_z
        else:
            ext_z = self.hp_init

        if self.lb is None:
            self.lb = ext_lb
        else:
            ext_lb = self.lb

        if self.ub is None:
            self.ub = ext_ub
        else:
            ext_ub = self.ub

        if self.is_integer is None:
            self.is_integer = ext_is_integer
        else:
            ext_is_integer = self.is_integer

        if self.step is None:
            self.step = ext_step
        else:
            ext_step = self.step

        if self.init_int_step is None:
            self.init_int_step = init_int_step
        else:
            init_int_step = self.init_int_step

        if self.base is None:
            self.base = base_exp
        else:
            base_exp = self.base

        # Creates the meshes of the grid where the hyperparameters must be
        # moved
        self.space_points, ext_z, ext_lb, ext_ub, ext_is_integer = self.create_meshes(
            self.n, self.hp_init, self.lb, self.ub, self.is_integer, self.base)

        # setting the common parameters in Fortran
        dfl.ext.ext_alfa_stop = self.alfa_stop
        dfl.ext.ext_nf_max = self.nf_max
        dfl.ext.ext_iprint = self.iprint

        # check if the variable is finite
        for i in range(self.n):
            if not numpy.isfinite(ext_z[i]):
                ext_z[i] = ext_lb[i]
        # check if the variables are in bound
        for i in range(self.n):
            if ext_z[i] > ext_ub[i]:
                raise ValueError('variable %s out of upper bound' % i)
            elif ext_z[i] < ext_lb[i]:
                raise ValueError('variable %s out of lower bound' % i)

        # Calls the optimization method
        result_z = dfl.main_box_discr(
            self.model_funct,
            ext_z,
            ext_lb,
            ext_ub,
            ext_step,
            init_int_step,
            ext_is_integer,
            self.n)

        # Final training with all optimal hyperparameters and all the samples
        # in the training set
        self.estimator.fit(self.X_values, self.Y_values)

        return self

    def predict(self, X):
        """
        Standard predict method for a Scikit-learn estimator
        """

        return self.estimator.predict(X)

    def create_meshes(self, n, ext_z, ext_lb, ext_ub,
                      ext_is_integer, base_exp):
        """
        Method that creates the meshes where the hyperparameters must be moved

        Note if on_a_mesh is 0 the chosen variables does not move in the
        meshes but rather in the space or real numbers if it is continuous or
        in the space of integer numbers if it is discrete all according to the
        list is_integer.
        """

        if self.bin_num is None:
            self.bin_num = [self.bin_num] * self.n

        space_points = [1] * self.n
        for i in range(self.n):
            if not self.on_a_mesh[i]:
                space_points[i] = 'continous'
            else:
                if self.hp_mesh_bin[i] is 'linear':
                    if self.bin_num[i] is None:
                        self.on_a_mesh[1] = 0
                        self.is_integer[1] = 1
                        continue
                    space_points[i] = numpy.linspace(
                        ext_lb[i], ext_ub[i], num=self.bin_num[i])
                elif self.hp_mesh_bin[i] is 'log':
                    if self.bin_num[i] is None:
                        self.bin_num[i] = int(ext_ub[i] - ext_lb[i] + 1)
                        ext_z[i] = base_exp**ext_z[i]
                    space_points[i] = numpy.logspace(
                        ext_lb[i], ext_ub[i], num=self.bin_num[i], base=base_exp)
                ext_lb[i] = 0
                ext_ub[i] = self.bin_num[i] - 1
                ext_is_integer[i] = 1
                if ext_z[i] not in space_points[i]:
                    temp = [0] * self.bin_num[i]
                    for j in range(self.bin_num[i]):
                        temp[j] = (
                            numpy.absolute(
                                ext_z[i] -
                                space_points[i][j]))
                    index_ = temp.index(min(temp))
                    ext_z[i] = index_
                    print(
                        'variable %s not on the grid, starting from the '
                        'closest point instead' %
                        i)
                else:
                    ext_z[i] = space_points[i].tolist().index(ext_z[i])

        return space_points, ext_z, ext_lb, ext_ub, ext_is_integer

    def model_funct(self, z, n):
        """
        function that is called directly by dfl in Fortran
        """

        print(z)
        self.interface_funct(z)
        return self.KFold_error()

    def interface_funct(self, z):
        """
        function that changes the values of the hyperparameters in the estimator

        Note that because of the peculiarity of the multilayer neural networks,
        a custom method has been created.
        This method can be a template for the creation of custom functions to
        change the values of the hyperparameters.
        """

        dict_ = {}

        y = [0] * len(z)
        if(isinstance(self.estimator, sklearn.neural_network.MLPRegressor) or
           isinstance(self.estimator, sklearn.neural_network.MLPClassifier)):
            dict_['hidden_layer_sizes'] = tuple(
                z[0:len(z) - 1].astype(int))
            y[0] = tuple(z[0:len(z) - 1].astype(int))
            if not self.on_a_mesh[len(z) - 1]:
                dict_['alpha'] = z[len(z) - 1]
                y[1] = z[len(z) - 1]
            else:
                dict_['alpha'] = self.space_points[len(
                    z) - 1][int(z[len(z) - 1])]
                y[1] = self.space_points[len(z) - 1][int(z[len(z) - 1])]
        else:
            for i in range(len(z)):
                if not self.on_a_mesh[i]:
                    if self.var_is_int[i]:
                        dict_[self.hp_list[i]] = int(z[i])
                    else:
                        dict_[self.hp_list[i]] = z[i]
                else:
                    if self.var_is_int[i]:
                        dict_[self.hp_list[i]] = int(
                            self.space_points[i][int(z[i])])
                        y[i] = int(self.space_points[i][int(z[i])])
                    else:
                        dict_[self.hp_list[i]
                              ] = self.space_points[i][int(z[i])]
                        y[i] = self.space_points[i][int(z[i])]

        # print(y)

        self.estimator.set_params(**dict_)

    def KFold_error(self):
        """
        Method that calculates the k-fold error on the training
        """

        # Get the splits
        kf = KFold(n_splits=self.kf_splits)
        kf.get_n_splits(self.X_values)
        error_fold = []

        for train_index, validation_index in kf.split(self.X_values):
            # Create the splits
            X_train = self.X_values[train_index] 
            X_validation = self.X_values[validation_index]
            Y_train = self.Y_values[train_index] 
            Y_validation = self.Y_values[validation_index]
            try:
                # Check if the values of the  hyperparameters are
                # alright
                self.estimator.fit(X_train, Y_train)
            except ValueError as e:
                print(
                    "The following error on the values of the hyperparameters "
                    "has occurred \"%s\", returning infinity" %e)
                return numpy.inf
            F_validation = self.estimator.predict(X_validation)
            try:
                # Check if the metrics is alright
                error = getattr(
                    sklearn.metrics,
                    self.metric)(
                    Y_validation,
                    F_validation)
            except NameError:
                print(
                    "The specified measure of error is not in the module "
                    "sklearn.metrics, returning the value of the mean "
                    "squared error")
                error = sklearn.metrics.mean_squared_error(
                    Y_validation, F_validation)
            error_fold.append(error)
        print('error:', numpy.mean(error_fold))
        if self.minimization:
            return numpy.mean(error_fold)
        else:
            return -numpy.mean(error_fold)
