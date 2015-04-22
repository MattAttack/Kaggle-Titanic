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
Best model:
GradientBoostingClassifier with accuracy 84.268%, and params:
        warm_start:     False
        loss:   exponential
        verbose:        0
        subsample:      1.0
        max_leaf_nodes: None
        learning_rate:  0.1
        min_samples_leaf:       1
        n_estimators:   100
        min_samples_split:      2
        init:   None
        min_weight_fraction_leaf:       0.0
        random_state:   None
        max_features:   None
        max_depth:      5
```

### NEXT
As a point of fact, submitting the output of this scored 0.77033.  Might have to dig into sklearns cross validation, or actually
use a separate test set to figure out how the model might generalize (this seems like a big drop).

### Goals:
- [x] high classification accuracy (above .8)
- [x] effective feature calculation
- [x] explore other approaches besides Random Forest.
