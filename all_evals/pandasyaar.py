import pandas as pd

df = pd.read_csv("./navyresults.csv")
df1 = pd.read_csv("./deepseek_analysis_results.csv")
print(df.head())


for i in range(5):

    print(df["ft_true_disease"][i], df1["ft_true_disease"][i])
