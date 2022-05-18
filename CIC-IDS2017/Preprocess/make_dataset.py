import pandas as pd
import sys
import glob
sys.path.append('..')
import Preprocess.csv_utils as cu
import Preprocess.meta_data_2018 as md8
import Preprocess.meta_data_2017 as md7

#根据标签分割数据集
def split_datasets_by_label(df, data_name, dst_path):
    if data_name == "ids2018":
        for label in md8.LABEL_LIST:
            print("***** Handling " + str(label) + " *******")
            td = cu.split_dataset_by_label(df, [label])
            cu.write_to_csv(td, dst_path + str(label) + ".csv")
    else:
        pass

#根据标签分割的数据集,构建子集
def make_subsets_by_ratio(file_path, ratio):
    all_data = []
    for f in glob.glob(file_path + "*.csv"):
        print("***** Processing " + str(f) + " *******")
        df = pd.read_csv(f, low_memory=False)
        td, td2 = cu.split_dataset_by_ratio(df, ratio)
        all_data.append(td)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data


#根据数量分割的数据集,构建子集
def make_subsets_by_num(file_path, num):
    all_data = []
    for f in glob.glob(file_path + "*.csv"):
        print("***** Processing " + str(f) + " *******")
        df = pd.read_csv(f, low_memory=False,encoding='ISO-8859-1')
        if df.shape[0]>num:
            all_data.append(df[:num])
        else:
            all_data.append(df)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data


#转换数据类型
def convert_datatype(df, data_name):
    if data_name == "ids2018":
        for row in md8.FEATURE_LIST[0:1]:
            df[row] = df[row].astype(float)
        for row in md8.FEATURE_LIST[2:-1]:
            df[row] = df[row].astype(float)
    if data_name == "ids2017":
        for row in md7.FEATURE_LIST[0:1]:
            df[row] = df[row].astype(float)
        for row in md7.FEATURE_LIST[2:-1]:
            df[row] = df[row].astype(float)            
    return df

#标签编码
def label_encoding(df, labels, option, data_name):
    if data_name == "ids2018":
        #小类编码
        if option == 1:
            codes = [ i for i in range(len(labels))]

        #大类编码
        elif option == 2:
            codes = [0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 4, 4, 6, 6]

     #二分编码
        elif option == 3:
            codes = []
            for label in labels:
                if label == "Benign":
                    codes.append(0)
                else:
                    codes.append(1)

        df['Label'].replace(labels, codes, inplace=True)
    return df

def main():
    file_path = "../split_csv/"
    dst_path = "../percent/"
    df = make_subsets_by_num(file_path, 150000)
    cu.write_to_csv(df, dst_path+"percent.csv")

if __name__ == "__main__":
    main()