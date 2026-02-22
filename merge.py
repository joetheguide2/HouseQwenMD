import numpy as np
import pandas as pd

df = pd.read_csv("./train1_with_json.csv")

df2 = pd.read_csv("./train2.csv")
df2 = df2.drop_duplicates(subset='messages', keep='first')
df3 = pd.concat([df[['messages']], df2[['messages']]], ignore_index=True)
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)
df3.to_csv("merged.csv")
print(df3.info())
