import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

df = pd.read_csv("Dataset_Hadi_Expanded.csv")

df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)

X = df[["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
        "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

if len(y_test) < 2:
    print("Error: Test set has less than two samples. Increase test size or dataset size.")
else:
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)

    ridge_model = Ridge(alpha=0.1)
    ridge_model.fit(X_train_poly, y_train)

    y_pred = ridge_model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)

    print(f"R^2 Score on Test Data: {r2:.4f}")
