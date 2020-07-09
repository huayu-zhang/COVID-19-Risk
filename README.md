# External validation the DL-death and DL-poor-outcome models

### Clone the repo
```shell script
git clone https://github.com/huayu-zhang/COVID-19-Risk.git
cd COVID-19-Risk
```

### Replace feature and outcome files in the repo folder with real data

**X.csv** with columns [index, age, male, neutrophil, lymphocyte, platelet, crp, creatinine]

**y_death.csv** with columns [index, death]

**y_poor_outcome.csv** with columns [index, poor_outcome] (poor outcome defined by OR(ICU, Death))

Column names do not have to be exact as long as the content is correct.


### Validation
```shell script
# Create and activiate the environment with conda
bash create_env.sh
conda activate covid_19_risk

# External validation
bash external_validation.sh
```

Results are saved at **validation_output/**

#
#
#
# Quick start for modeling

### Setup
```python
import sys; print('Python %s on %s' % (sys.version, sys.platform))
import os
sys.path.extend(os.getcwd())
```

### Data preparation
X is dataframe (or array) with n features and m observations with shape (m, n)

y is dataseries (or array) with m observations with shape (m, )


### Fitting basic LR regression model
```python
from sklearn.linear_model import LogisticRegression
from utils.modelling_methods.logistic_regression import SklearnBinaryClassification

params = {
    'C': 1,
    'max_iter': 5000,
    'penalty': 'l1',
    'solver': 'liblinear'
}

estimator = LogisticRegression()
lr_model = SklearnBinaryClassification(estimator=estimator)
lr_model.set_params(**params)
lr_model.fit(X=X, y=y, X_test=None, y_test=None) # If X_test and y_test also provided metrics reported on test set, 
                                                 # otherwise on train set. If you want estimator, then train using all data

print(lr_model.get_params())
print(lr_model.coef_dict)
print(lr_model.log)
print(lr_model.get_estimator()) # How to get estimator
lr_model.save_estimator(path='estimator_example') # How to save estimator
```

### Cross-validation of LR model
```python
from utils.modelling_methods.logistic_regression import LRCrossValidation

params = {
    'C': 1,
    'max_iter': 5000,
    'penalty': 'l1',
    'solver': 'liblinear'
}

lr_cv = LRCrossValidation(X=X, y=y, n_jobs=None) # n_jobs=None use all cores for parallel cv, if not linux system, 
                                                 # set n_jobs = 1 for single core implementation
lr_cv.set_params(**params)
lr_cv.cross_validation()

print(lr_cv.get_params())
print(lr_cv.log)
print(lr_cv.cv_results_full)
```

### Tuning on range of params
```python
from utils.modelling_methods.logistic_regression import LRCrossValidation, GridTuning


params_range_l1 = {
    'C': [2**i for i in range(-7, 5)],
    'max_iter': 5000,
    'penalty': 'l1',
    'solver': 'liblinear'
}

cv = LRCrossValidation(X=X, y=y) # n_jobs=None use all cores for parallel cv
                                 #if not linux system, set n_jobs = 1 for single core implementation
tune = GridTuning(cv=cv, params_range=params_range_l1)

tune.tuning()

print(tune.tune_params)
print(tune.tune_results)
print(tune.get_best_performance())
print(tune.get_best_params())
print(tune.get_best_model())
print(tune.get_best_estimator()) # This function returns best estimator from tuning, using all data
```
