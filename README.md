## Kaggle Titanic
Repository for the [Kaggle Titanic problem](http://www.kaggle.com/c/titanic-gettingStarted) which is for learning purposes only.

Files in the R/ directory are for exploring the data set.  Run with
```
$ Rscript explore.R
```
We aren't using a virtualenv, because numpy/scipy/sklearn don't install cleanly, so installation will be a little 
fussy.  Probably if you are using [anacondas](https://store.continuum.io/cshop/anaconda/) you'll be fine.


Fit a model using
```
$ python model.py
```

There should be some diagnostic information printed from the cross validation stage, and a file should be printed
that is suitable for submission to kaggle.

### Goals:
[x] high classification accuracy (above .8)
[x] effective feature calculation
[ ] explore other approaches besides Random Forest.
