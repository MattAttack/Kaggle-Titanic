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

## NEXT
Expand the `model.get_model` function so that it runs through a list of models and parameters and automatically selects the best one.  For example, the list of models might start:
```
models = [ (LogisticRegression, {"penalty": ("l1", "l2"), "C": [10 ** (0.5 * j) for j in range(-10, 10)]}), ... ]
```
Then sklearn is nice, so we could do something like
```
for model_func, model_params in models:
  for parameter_dict in itertools.have_to_lookup_this_function(model_params):
    clf = model_func(**parameter_dict)
    ...
```
sklearn's api gives us that the object `clf` will have the methods `.fit`, `.transform`, `.score`, `.fit_transform`, so we should be able to continue with the cross validation already in there. Then we save the `model_func` and `parameter_dict` from the best model, and return those.

### Goals:
- [x] high classification accuracy (above .8)
- [x] effective feature calculation
- [ ] explore other approaches besides Random Forest.
