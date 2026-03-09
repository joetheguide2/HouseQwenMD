import pandas as pd
import json
import os

# ----------------------------------------------------------------------
# List of all CSV files (from the image)
# ----------------------------------------------------------------------
df = pd.read_csv("./ft6.csv")

df = df.drop(['Unnamed: 0.5', 'Unnamed: 0.4', "Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0"], axis=1)
df.to_csv("ft7.csv")
df.to_parquet("training.parquet")
print(df.info())
