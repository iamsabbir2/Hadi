import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset_Hadi.csv")


plt.figure(figsize=(8, 5))
sns.regplot(x="sqft_above", y="price", data=df)

plt.show()
