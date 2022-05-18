import pandas as pd
import glob

#读取csv数据集
def load_single_file_data(file_path, with_header):
    if with_header:
        df = pd.read_csv(file_path, low_memory=False)
    else:
        df = pd.read_csv(file_path, header=None, low_memory=False)
    return df

#读取csv数据集
def load_all_file_data(dir_path, with_header):
    all_data = []
    for f in glob.glob(dir_path + "*.csv"):
        if with_header:
            df = pd.read_csv(f, low_memory=False)
            all_data.append(df)
        else:
            df = pd.read_csv(f, header=None, low_memory=False)
            all_data.append(df)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data

#属性值筛选
def select_attribute(df, attr_list):
    df = df[attr_list]
    return df

#属性值筛选
def drop_attribute(df, attr_list):
    df = df.drop(attr_list, axis=1)
    return df

#根据类别划分
def split_dataset_by_label(df, label_list):
    all_data = []
    for label in label_list:
        td = df[(df['Label'].str.contains(label))]
        all_data.append(td)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data

#根据比例划分
def split_dataset_by_ratio(df, ratio):
    num_split = int(df.shape[0] * ratio)
    return df[:num_split], df[num_split:]

#根据数目划分
def split_dataset_by_num(df, num):
    #num_split = int(df.shape[0] * ratio)
    return df[:num], df[num:]

#输出到csv
def write_to_csv(df, file_name):
    pd.DataFrame(df).to_csv(file_name, index=None)