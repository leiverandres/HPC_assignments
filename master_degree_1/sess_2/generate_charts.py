import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("./seq_times.csv")
df1 = df1.set_index("size")
df1 = df1.groupby("size", as_index=False).mean()

df2 = pd.read_csv("./opt_times_miss_cache.csv")
df2 = df2.set_index("size")
df2 = df2.groupby("size", as_index=False).mean()
print(df2.head())
x = df1.plot(kind="line)
df2.plot(kind="line", ax=x)
plt.show()
