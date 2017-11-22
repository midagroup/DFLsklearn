# DFLsklearn

DFLsklearn is a method to find the hyperparameters of a machine learning algorithm in [scikit-learn](http://scikit-learn.org/). 
The method automatically supports many estimators among the ones present in Sklearn, but it is also capable of handling all the algorithms that inherit from the ```BaseEstimator``` class of sklearn.

DFLsklearn has been tested on linux and macos distributions.

## Installation

    The code can be downloaded from github:
    git clone git@github.com:midagroup/DFLsklearn.git

and the installation can be performed in the root of the downloaded folder by using the standard pip command:

    Python setup.py install

## Usage

The hyperparameters search can be perfomed by choosing which sklearn estimator use and then add a single command line to call the DFLsklearn optimization:
```
import sklearn
import DFLscikit 

#load data
#Choose the estimator


estimator_path='sklearn.neural_network.MLPRegressor'

model=DFLscikit.DFL_estimator(estimator_path=estimator_path)

model.fit(X_train,Y_train)

print(sklearn.metrics.accuracy_score(model.predict(X_test),Y_test))
```

Complete example using the Diabetes dataset:
```
import numpy
import sklearn
import DFLscikit 


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
#Create the DFLscikit object and train
#------------------------------------------------------------------------------ 
    
    estimator_path='sklearn.neural_network.MLPRegressor'
    model=DFLscikit.DFL_estimator(estimator_path=estimator_path)
    model.fit(X_train,Y_train)
    print('Optimization complete, coefficient of determination on the test set: 
    %f'% sklearn.metrics.r2_score(model.predict(X_test),Y_test))
```            
DFLsklearn also offers the possibility to personalize the optimization by choosing the hyperparameters of the estimator, the lower and upper bounds in which such parameters should be optimized and other more advanced options to speed up the optimization process.

For further examples see  the code in the examples.py. 
The file examples.py has examples both for classification and regression, plus an example to initialize a custom estimator from sklearn.

## Out of the box optimization
DFLsklearn comes with a built in method called ```configuration_preset``` that sets the different optimization options  according to pre-specified default values that allow to optimize the estimator in a reasonably large neighborhood of the default Scikit-learn hyperparameter values. The ```configuration_preset``` is compatible with the following Sklearn estimators:
```
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
```
However it is possible to set up the optimization in two alternative ways:
- Pass the desired arguments as formal arguments when initializing the DFF_estimator object
```
    model=DFLscikit.DFL_estimator(estimator_path=estimator_path,
                                  preset_config=0,base=10.0,
                                  hp_list=['hidden_layer_sizes','alpha'], 
                                  hp_init=numpy.array([(10,10,),-8]), 
                                  lb=numpy.array([10,10,-8]), 
                                  ub=numpy.array([100,100,8]), 
                                  step=[1,1,1], 
                                  init_int_step=[10,10,1],
                                  is_integer=[1,1,0],
                                  on_a_mesh=[0,0,1],
                                  var_is_int=[0,0,0],
                                  estimator_param={'random_state':1},
                                  iprint=0)    
```

- Pass a pointer to a function with the same structure as ```configuration_preset```  written by the user:
```
    model=DFLscikit.DFL_estimator(estimator_path=estimator_path,
                                  preset_function=newpreset)
    #...
    def newpreset(model_f):
        n=3
        base=10.0
        hp_list=['hidden_layer_sizes','alpha']
        hp_init=[(10,10,),-8]
        lb=numpy.array([10,10,-8])
        ub=numpy.array([100,100,8]) 
        step=[1,1,1]
        init_int_step=[10,10,1]
        is_integer=[1,1,0]
        on_a_mesh=[0,0,1]
        var_is_int=[0,0,0]

        return n,hp_init,lb,ub ,step,init_int_step,is_integer,base,hp_list,on_a_mesh,var_is_int

```
In this way DFLsklearn supports both custom optimization by expert users and the optimization of custom estimators that satisfy the contributing guidelines of Scikit-learn http://scikit-learn.org/stable/developers/contributing.html.

# License
The software is distribuited under the New BSD License. Check the License.md file for more information.

# Others
The file "custom_funct_interfaces.py" contains some examples of custom functions that interface DFL to the estimators of sklearn. Such functions can be used as templates for a user made functions.
