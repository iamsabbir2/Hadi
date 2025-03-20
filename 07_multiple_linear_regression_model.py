from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("Dataset_Hadi.csv")


# Define feature set
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
y = df["price"]

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Calculate R^2
r2_score = model.score(X, y)
print("R^2 Score:", r2_score)
