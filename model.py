import os
import itertools
from sklearn.ensemble import RandomForestClassifier
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
models = [
    (LogisticRegression, {"penalty": ("l1", "l2"), "C": [10 ** (0.5 * j) for j in range(-1, 1)]}),
    (GaussianNB, {}),
    (RandomForestClassifier, {}),
    (SVC, {})
]


def get_possible_params(parameter_dict):
    for p in itertools.product(*parameter_dict.values()):
        yield dict(zip(parameter_dict.keys(), p))


def get_model():
    features, labels, _ = FEATURE_GETTER.features_labels_and_ids('train')
    best_score = 0
    best_parms = {}

    for model, param_opts in models:
        for params in get_possible_params(param_opts):
            clf = model(**params)
            score = cross_val_score(clf, features, labels, cv=20).mean()
            if score > best_score:
                best_model = model
                best_score = score
                best_parms = params

    final_model = best_model(**best_parms)
    final_model.fit(features, labels)

    print("Best model was {:s} with {:s}, accuracy {:.3f}%, and params:\n\t{:s}".format(
        model.__name__,
        ", ".join(["{:s} {:s}".format(k, str(v)) for k, v in best_parms.iteritems()]),
        100 * best_score,
        "\n\t".join(
            ["{:s}:\t{}".format(*j) for j in final_model.get_params().iteritems()]
        )
    ))
    return final_model


def write_predictions():
    clf = get_model()
    features, _, ids = FEATURE_GETTER.features_labels_and_ids('test')
    predictions = clf.predict(features)
    fname = os.path.join(utils.OUT_DATA, 'predictions.csv')
    with open(fname, 'wb') as buff:
        buff.write("PassengerId,Survived\n")
        buff.write("\n".join(["{:d},{:d}".format(*j) for j in zip(ids, predictions)]))


def main():
    write_predictions()


if __name__ == '__main__':
    main()
