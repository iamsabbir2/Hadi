import pandas as pd

df = pd.read_csv("Dataset_Hadi.csv")

floor_counts = df["floors"].value_counts().to_frame()

print(floor_counts)
