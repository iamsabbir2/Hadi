from sklearn.linear_model import Ridge
import pandas as pd

df = pd.read_csv("Dataset_Hadi.csv")

features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]
y = df["price"]


ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X, y)

r2_score = ridge_model.score(X, y)
print("R^2 Score:", r2_score)
