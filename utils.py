import os
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')


def get_features(data_set='train', return_type='array'):
    fname = TRAIN_CSV if data_set == 'train' else TEST_CSV
    with open(fname, 'r') as buff:
        if return_type == 'dict':
            reader = csv.DictReader(buff)
        else:
            reader = csv.reader(buff)
            next(reader)  # skip header
        data = list(reader)
    return data


def main():
    print(get_features(return_type='dict')[:10])


if __name__ == '__main__':
    main()
