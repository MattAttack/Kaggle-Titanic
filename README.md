## Kaggle Titanic
Repository for the [Kaggle Titanic problem](http://www.kaggle.com/c/titanic-gettingStarted) which is for learning purposes only.

Files in the R/ directory are for exploring the data set.  Run with
```
$ Rscript explore.R
```
We aren't using a virtualenv, because numpy/scipy/sklearn don't install cleanly, so installation will be a little 
fussy.  Probably if you are using [anacondas](https://store.continuum.io/cshop/anaconda/) you'll be fine.

### Goals:
- high classification accuracy (above .8)
- effective feature calculation
- explore other approaches besides Random Forest.
