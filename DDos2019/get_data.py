import numpy as np
import pandas as pd
import Preprocess.csv_utils as cu
import Preprocess.make_dataset as md
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.append('/opt/anaconda3/lib/python3.7/site-packages/tensorflow/examples/tutorials/mnist')
sys.path.append('..')


def prepare_2019(source_dir, save_data_dir, save_test_data_dir, train_rate):
    """随机化数据 shuffle"""
    df = pd.read_csv(source_dir, low_memory=False)
    df = shuffle(df)
    print(df.columns.values)
    '''去除无用的特征 drop'''
    df = cu.drop_attribute(df, ['Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port', ' Destination IP',
                                ' Destination Port', ' Protocol', ' Timestamp', 'SimillarHTTP'])
    df = md.convert_datatype(df, "ddos2019")
    print(df.shape)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)
    print(df.shape)
    for s in df.std():
        print(s)
    df.drop(df.columns[:-1][df.std() == 0], axis=1, inplace=True)
    print(df.shape)
    print(df.columns.values)
    '''归一化normalization'''
    values = df.values
    images, labels = values[:, :-1], values[:, -1]
    images = preprocessing.MinMaxScaler().fit_transform(images)
    all_classes = np.unique(labels)
    print(df.shape, all_classes)
    '''统计和保存训练文件'''
    train_num = int(len(labels) * train_rate)
    for i in range(len(all_classes)):
        print('num of class ', i, all_classes[i], np.sum(labels == all_classes[i]))
        print('num of train class ', i, all_classes[i], np.sum(labels[:train_num] == all_classes[i]))
        print('num of test class ', i, all_classes[i], np.sum(labels[train_num:] == all_classes[i]))

    np.save(save_data_dir,
            np.concatenate((images[:train_num], np.expand_dims(labels[:train_num], axis=1)), axis=1))
    np.save(save_test_data_dir,
            np.concatenate((images[train_num:], np.expand_dims(labels[train_num:], axis=1)), axis=1))
    return all_classes


def get_train_test(unknown_class, DATA_DIR, TEST_DATA_DIR):
    print('--starting get data sets--')

    print('--getting  test data--')
    test_data = np.load(TEST_DATA_DIR, allow_pickle=True)
    for i in range(len(test_data)):
        if test_data[i, -1] in unknown_class:
            test_data[i, -1] = 'unknown'
    all_classes = np.unique(test_data[:, -1])
    LE = LabelEncoder()
    LE.fit(all_classes)
    test_label = LE.transform(test_data[:, -1])
    known_test_data = np.concatenate((test_data[:, :-1].astype(float),
                                      np.expand_dims(test_label, axis=1)
                                      ), axis=1)

    print('--getting  training data--')
    train_data = np.load(DATA_DIR, allow_pickle=True)
    known_train_data = np.array([image for image in train_data if image[-1] not in unknown_class])

    known_class = np.unique(known_train_data[:, -1])
    known_train_label = LE.transform(known_train_data[:, -1])
    known_train_data = np.concatenate((known_train_data[:, :-1].astype(float),
                                       np.expand_dims(known_train_label, axis=1)), axis=1)

    print('--ending get dataset', 'training data shape:', known_train_data.shape, 'testing data shape:',
          known_test_data.shape)
    return known_train_data, known_test_data, known_class


if __name__ == '__main__':
    SOURCE_DIR = 'percent/percent.csv'
    SAVE_DATA_DIR = 'osd/ddos2019_train_data.npy'
    SAVE_TEST_DATA_DIR = 'osd/ddos2019_test_data.npy'
    data_train_rate = 0.8
    this_labels = prepare_2019(SOURCE_DIR, SAVE_DATA_DIR, SAVE_TEST_DATA_DIR, data_train_rate)
