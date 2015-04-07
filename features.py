import numpy
from sklearn.preprocessing import StandardScaler

import utils


def gender_func(row, **kwargs):
    return 0 if row["Sex"] == 'male' else 1


def age_func(row, **kwargs):
    data = kwargs["data"]
    age = row["Age"]
    if age != '':
        return float(age)
    return numpy.mean([float(j["Age"]) for j in data if j["Age"] != ''])


def sibling_func(row, **kwargs):
    return float(row["SibSp"])


class Features:
    feature_transforms = [
        gender_func,
        age_func,
        sibling_func,
    ]
    feature_labels = [
        "gender",
        "age",
        "siblings"
    ]

    def __init__(self):
        self._train = None
        self._test = None
        self.scaler = StandardScaler()
        self._is_scaled = False

    def data(self, data_set):
        if data_set == 'train':
            return self.train
        return self.test

    @property
    def train(self):
        if self._train is None:
            self._train = utils.get_features(data_set='train', return_type='dict')
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = utils.get_features(data_set='test', return_type='dict')
        return self._test

    def labels(self, data_set='train'):
        return numpy.array([int(j.get("Survived", 0)) for j in self.data(data_set)])

    def raw_features(self, data_set='train'):
        return numpy.array(
            [[func(j, data=self.data(data_set)) for func in self.feature_transforms] for j in self.data(data_set)]
        )

    def features(self, data_set='train'):
        if not self._is_scaled:
            self.scaler.fit(self.raw_features('train'))  # only fit on the training set
            self._is_scaled = True
        return self.scaler.transform(self.raw_features(data_set))

    def ids(self, data_set='train'):
        return [j["PassengerId"] for j in self.data(data_set)]

    def features_labels_and_ids(self, data_set='train'):
        return self.features(data_set), self.labels(data_set), self.ids(data_set)
