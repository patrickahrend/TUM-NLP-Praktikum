import pandas as pd 



df_1 = pd.read_csv("../data/1_Matching_Labels.csv", encoding="utf-8")
df_2 = pd.read_csv("../data/2_Matching_Labels.csv", encoding="utf-8")


print(df_1['label'].value_counts())
print(df_2['label'].value_counts())
