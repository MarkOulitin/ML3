import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split


def idxToCsv(idx, name):
    df = pd.DataFrame(data=idx, columns=['dataset_index'])
    df.to_csv(f'{name}_idx.csv')


def partitionToIdx():
    idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.4).split(np.arange(8189)))
    train_idx, validation_idx = train_test_split(idx, test_size=0.5)
    idxToCsv(train_idx, 'train')
    idxToCsv(validation_idx, 'validation')
    idxToCsv(test_idx, 'test')


def readIdx(file):
    df = pd.read_csv(file)
    idx = np.array(df.iloc[:, 1])
    return idx


def main():
    pass


if __name__ == '__main__':
    main()
