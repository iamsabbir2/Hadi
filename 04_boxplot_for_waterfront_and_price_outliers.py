import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Dataset_Hadi.csv")

plt.figure(figsize=(8, 5))
sns.boxplot(x="waterfront", y="price", data=df)

plt.show()
