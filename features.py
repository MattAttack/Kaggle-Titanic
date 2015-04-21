import sys
import numpy
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import utils


class Features:
    categories = [
        "Pclass",
        "Embarked"
    ]

    def __init__(self):
        self._train = None
        self._test = None
        self.scaler = StandardScaler()
        self._labels = {}  # to encode categorical variables, we use LabelEncoder to turn columns into integers,
        self._raw_features = {}
        self.enc = OneHotEncoder(sparse=False)  # then OneHotEncoder to turn integers into binary arrays.
        # For example "Embarked C" --> 1 --> [0, 1, 0].
        self._is_scaled = False
        self._is_encoded = False
        self._means = {}

    def category_labels(self):
        labels = []
        for category in self.categories:
            for j in self.labelencoder(category).classes_:
                labels.append("{:s} {}".format(category, j))
        return labels

    def feature_labels(self):
        return ["gender",
                "age",
                "siblings and spouses",
                "parents and children",
                "fare"] + self.category_labels()

    @property
    def feature_funcs(self):
        return [self.gender_func,
                self.float_col("Age"),
                self.float_col("SibSp"),
                self.float_col("Parch"),
                self.float_col("Fare"),
                self.category_cols
                ]

    def _encode(self):
        if not self._is_encoded:
            self._is_encoded = True

            for cat in self.categories:
                self.train[cat] = self.labelencoder(cat).transform(self.train[cat].values)

            self.enc.fit(self.train[self.categories].values)


    def labelencoder(self, col):
        if col not in self._labels:
            self._labels[col] = LabelEncoder().fit(self.train[col].values)
        return self._labels[col]

    def label_col(self, row, col):
        return self.labelencoder(col).transform(row[col])

    def mean_col(self, col):
        if col not in self._means:
            self._means[col] = numpy.mean([float(j[col]) for (idx, j) in self.train.iterrows() if j[col]])
        return self._means[col]

    def float_col(self, col):
        def func(row):
            try:
                return [float(row[col])]
            except ValueError:
                return [self.mean_col(col)]

        return func

    def category_cols(self, row):
        self._encode()
        try:
            val = [self.label_col(row, cat) for cat in self.categories]
            return self.enc.transform([val]).tolist()[0]
        except ValueError, e:
            print '\n\n*** ERROR: caught value error', e, '***\n\n'
            print 'row:\n', row
            sys.exit(1)

    def gender_func(self, row):
        return [int(row["Sex"] != 'male')]

    def data(self, data_set):
        if data_set == 'train':
            return self.train
        return self.test

    @property
    def train(self):
        if self._train is None:
            self._train = utils.get_features(data_set='train')
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = utils.get_features(data_set='test')
        return self._test

    def labels(self, data_set='train'):
        if data_set == 'train':
            return numpy.array([int(j["Survived"]) for (idx, j) in self.data(data_set).iterrows()])

    def raw_features(self, data_set='train'):
        if not self._raw_features.has_key(data_set):
            self._raw_features[data_set] = numpy.array(
                [sum([func(j) for func in self.feature_funcs], []) for (idx, j) in self.data(data_set).iterrows()]
            )
        return self._raw_features[data_set]

    def features(self, data_set='train'):
        if not self._is_scaled:
            self.scaler.fit(self.raw_features('train'))  # only fit on the training set
            self._is_scaled = True

        return self.scaler.transform(self.raw_features(data_set))

    def ids(self, data_set='train'):
        return [j["PassengerId"] for (idx, j) in self.data(data_set).iterrows()]

    def features_labels_and_ids(self, data_set='train'):
        return self.features(data_set), self.labels(data_set), self.ids(data_set)


def main():
    print(Features().category_labels())


if __name__ == '__main__':
    main()
