import pandas as pd

df1 = pd.read_csv("split_csv_new/Benign1.csv", low_memory=False)
df2 = pd.read_csv("split_csv_new/Benign2.csv", low_memory=False)
print(df1.shape)
print(df2.shape)
df =  pd.concat([df1, df2], axis=0, ignore_index=True)
print(df.shape)
pd.DataFrame(df).to_csv("split_csv_new/Benign.csv", index=None)