import pandas as pd
import sys

#处理缺失值    
def handle_missing(df, option):
    missing_data_index = []
    for rows in df:
        if rows == "Label":
            continue
        if df[rows].isnull().sum() != 0:
            missing_data_index.append(rows)

    #删除属性值
    if option == 1:
        df = df.drop(missing_data_index, axis=1)

    #0值填充缺失值
    elif option == 2:
        for row in missing_data_index:
            df[row].fillna(0)
    
    #均值填充缺失值
    elif option == 3:
        for row in missing_data_index:
            mean = df[row].mean()
            df[row] = df[row].astype(float)
            df[row].fillna(mean)
    
    #中位数填充缺失值
    elif option == 4:
        for row in missing_data_index:
            median = df[row].median()
            df[row].fillna(median)

    return df