import pandas as pd
import glob


# 读取数据集
def load_single_file_data(file_path, with_header):
    if with_header:
        df = pd.read_csv(file_path, low_memory=False)
    else:
        df = pd.read_csv(file_path, header=None, low_memory=False)
    return df


# 读取数据集
def load_all_file_data(dir_path, with_header):
    all_data = []
    for f in glob.glob(dir_path + "*.csv"):
        print(f)
        if with_header:
            df = pd.read_csv(f, low_memory=False)
            all_data.append(df)
        else:
            df = pd.read_csv(f, header=None, low_memory=False)
            all_data.append(df)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data


# 属性值筛选
def select_attribute(df, attr_list):
    df = df[attr_list]
    return df


# 属性值筛选
def drop_attribute(df, attr_list):
    df = df.drop(attr_list, axis=1)
    return df


# 根据类别划分
def split_dataset(df, label_list):
    all_data = []
    for label in label_list:
        # print((df['Label'].str.contains(label)))
        td = df[(df[' Label'].str.contains(label))]
        all_data.append(td)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data


# 处理缺失值
def handle_missing(df, option):
    missing_data_index = []
    for rows in df:
        if rows == "Label":
            continue
        if df[rows].isnull().sum() != 0:
            missing_data_index.append(rows)

    # 删除属性值
    if option == 1:
        df = drop_attribute(df, missing_data_index)

    # 0值填充缺失值
    elif option == 2:
        for row in missing_data_index:
            df[row].fillna(0)

    # 均值填充缺失值
    elif option == 3:
        for row in missing_data_index:
            mean = df[row].mean()
            df[row] = df[row].astype(float)
            df[row].fillna(mean)

    # 中位数填充缺失值
    elif option == 4:
        for row in missing_data_index:
            median = df[row].median()
            df[row].fillna(median)

    return df


# 标签编码
def label_encoding(df, labels, option):
    # 小类编码
    if option == 1:
        codes = [i for i in range(len(labels))]

    # 大类编码
    elif option == 2:
        codes = [0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 4, 4, 6, 6]

    # 二分编码
    elif option == 3:
        codes = []
        for label in labels:
            if label == "Benign":
                codes.append(0)
            else:
                codes.append(1)

    df['Label'].replace(labels, codes, inplace=True)
    return df


# 输出到csv
def write_to_csv(df, file_name):
    pd.DataFrame(df).to_csv(file_name, index=None)


import numpy as np

if __name__ == "__main__":

    df = load_all_file_data("./MachineLearningCSV/", True)
    labels = df.values[:, -1]
    all_classes = np.unique(labels)
    for label in all_classes:
        print("*****Handling " + str(label) + ".....................")
        td = split_dataset(df, [label])

        write_to_csv(td, "./split_csv/" + str(label).replace("?", "_") + ".csv")
        print(td.shape)
    print(df.shape)
    print(df.columns)
    print(df.head())
