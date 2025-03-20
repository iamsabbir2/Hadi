from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("Dataset_Hadi.csv")

# Define X (feature) and y (target)
X = df[["sqft_living"]]
y = df["price"]

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Calculate R^2
r2_score = model.score(X, y)
print("R^2 Score:", r2_score)
