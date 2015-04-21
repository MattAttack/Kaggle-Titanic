import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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


def get_model():
    features, labels, _ = FEATURE_GETTER.features_labels_and_ids('train')
    print 'got features'
    best_score = 0
    best_parms = {}
    for penalty in ("l1", "l2"):
        for C in [10 ** (0.5 * j) for j in range(-1, 1)]:
            clf = LogisticRegression(penalty=penalty, C=C)
            #clf = GaussianNB()
            score = cross_val_score(clf, features, labels, cv=20).mean()
            if score > best_score:
                best_score = score
                best_parms = {"penalty": penalty, "C": C}
                
    best_model = LogisticRegression(**best_parms)
    best_model.fit(features, labels)

    print("Best model was logistic regression with {:s} regularization, constant {:s}, accuracy {:.3f}%, and coefficients:\n\t{:s}".format(
        best_parms["penalty"],
        str(best_parms["C"]),
        100 * best_score,
        "\n\t".join(
            ["{:s}:\t{:.3f}".format(*j) for j in zip(FEATURE_GETTER.feature_labels(), best_model.coef_[0])]
        )
    ))
    return best_model


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
