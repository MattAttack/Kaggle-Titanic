import os
import itertools
import numpy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

import utils
from features import Features

"""
I'm using a global variable for the features since there are some preprocessors (right now, just the StandardScaler)
that should be fit with the training data, and should only transform the testing data, otherwise there would be leakage
from one data set to the other.  This should perhaps be moved into a Model class, but right now, Features() has no
side effects (all data sets are loaded lazily).
"""
FEATURE_GETTER = Features()
MODELS = [
    (LogisticRegression, {"penalty": ("l1", "l2"), "C": [10 ** (0.5 * j) for j in range(-1, 1)]}),
    (GaussianNB, {}),
    (RandomForestClassifier, {"n_estimators": (100, 200), "max_depth": range(2, 5)}),
    (GradientBoostingClassifier, {"loss": ("deviance", "exponential"), "learning_rate": (0.001, 0.01, 0.1), "max_depth": range(4, 8)}),
    (SVC, {})
]


def get_possible_params(parameter_dict):
    for p in itertools.product(*parameter_dict.values()):
        yield dict(zip(parameter_dict.keys(), p))


def get_model(verbose=False):
    features, labels, _ = FEATURE_GETTER.features_labels_and_ids('train')
    best_score = 0
    best_parms = {}

    for model, param_opts in MODELS:
        for params in get_possible_params(param_opts):
            clf = model(**params)
            score = numpy.median(
                numpy.array(cross_val_score(clf, features, labels, cv=20))
            )
            if score > best_score:
                if verbose:
                    print("New record:\n{:s}".format(print_model(model, clf, score)))
                best_model = model
                best_score = score
                best_parms = params

    final_model = best_model(**best_parms)
    final_model.fit(features, labels)

    print("Best model:\n{:s}".format(print_model(best_model, final_model, best_score)))
    return final_model

def print_model(model_func, trained_model, score):
    return "{:s} with accuracy {:.3f}%, and params:\n\t{:s}".format(
        model_func.__name__,
        100 * score,
        "\n\t".join(
            ["{:s}:\t{}".format(*j) for j in trained_model.get_params().iteritems()]
        )
    )


def write_predictions(verbose=False):
    clf = get_model(verbose=verbose)
    features, _, ids = FEATURE_GETTER.features_labels_and_ids('test')
    predictions = clf.predict(features)
    fname = os.path.join(utils.OUT_DATA, 'predictions.csv')
    with open(fname, 'wb') as buff:
        buff.write("PassengerId,Survived\n")
        buff.write("\n".join(["{:d},{:d}".format(*j) for j in zip(ids, predictions)]))


def main():
    write_predictions(verbose=False)


if __name__ == '__main__':
    main()
