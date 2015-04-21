import os
import csv
import pandas as pd
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUT_DATA = os.path.join(DIR, 'out_data')

if not os.path.exists(OUT_DATA):
    os.mkdir(OUT_DATA)


def get_features(data_set='train', return_type='array'):
    fname = TRAIN_CSV if data_set == 'train' else TEST_CSV
    df = pd.read_csv(fname)
    df.ix[df['Embarked'].isnull(), 'Embarked'] = 'S'
    return df.replace(np.nan, 0)


def main():
    print(get_features(return_type='dict')[:10])


if __name__ == '__main__':
    main()
