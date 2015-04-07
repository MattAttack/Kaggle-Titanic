import numpy

import utils


def gender_func(row, **kwargs):
    return 0 if row["Sex"] == 'male' else 1


def age_func(row, **kwargs):
    data = kwargs["data"]
    age = row["Age"]
    if age != '':
        return float(age)
    return numpy.mean([float(j["Age"]) for j in data if j["Age"] != ''])


class Features:
    feature_transforms = [
        gender_func,
        age_func
    ]

    def __init__(self, data_set='train'):
        self.data_set = data_set
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = utils.get_features(data_set=self.data_set, return_type='dict')
        return self._data

    def labels(self):
        return numpy.array([int(j["Survived"]) for j in self.data])

    def features(self):
        return numpy.array([[func(j, data=self.data) for func in self.feature_transforms] for j in self.data])

    def features_and_labels(self):
        return self.features(), self.labels()
