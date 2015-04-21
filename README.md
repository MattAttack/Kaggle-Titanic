## Kaggle Titanic
Repository for the [Kaggle Titanic problem](http://www.kaggle.com/c/titanic-gettingStarted) which is for learning purposes only.

Files in the R/ directory are for exploring the data set.  Run with
```
$ Rscript explore.R
```
You should be able to run this in a virtual environment.  From the cloned repo, run
```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python model.py
```

There should be some diagnostic information printed from the cross validation stage, and a file should be printed
that is suitable for submission to kaggle.

### Current record
```
Best model was SVC with , accuracy 82.147%, and params:
    kernel: rbf
    C:  1.0
    verbose:    False
    probability:    False
    degree: 3
    shrinking:  True
    max_iter:   -1
    random_state:   None
    tol:    0.001
    cache_size: 200
    coef0:  0.0
    gamma:  0.0
    class_weight:   None
```


### Goals:
- [x] high classification accuracy (above .8)
- [x] effective feature calculation
- [x] explore other approaches besides Random Forest.
