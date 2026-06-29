import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_sas("BMX_J.xpt")
df.to_csv("BMX_J.csv", index=False)

df = df[[
    "BMXWT",
    "BMXHT",
    "BMXBMI",
    "BMXWAIST",
    "BMXHIP",
    "BMXARMC",
    "BMXARML",
    "BMXLEG"
]]

print(df.head())
print(df.describe())
print(df.info())

print("Original Dataset Shape:", df.shape)

df = df.dropna()
df = df.drop_duplicates()

print("Cleaned Dataset Shape:", df.shape)

df["Waist_Hip_Ratio"] = df["BMXWAIST"] / df["BMXHIP"]

print(df.corr())

print("\nCorrelation with BMI:")
print(df.corr()["BMXBMI"].sort_values(ascending=False))

plt.figure(figsize=(7,5))
plt.hist(df["BMXBMI"], bins=15, edgecolor="black")
plt.title("Distribution of BMI Among NHANES Participants")
plt.xlabel("BMI")
plt.ylabel("Number of Participants")
plt.show()

plt.figure(figsize=(7,5))
plt.hist(df["BMXHT"], bins=15, edgecolor="black")
plt.title("Distribution of Height")
plt.xlabel("Height (cm)")
plt.ylabel("Number of Participants")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(df["BMXHT"], df["BMXWT"])
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()

X = df[[
    "BMXWT",
    "BMXHT",
    "BMXWAIST",
    "BMXHIP",
    "BMXARMC",
    "BMXARML",
    "BMXLEG",
    "Waist_Hip_Ratio"
]]

y = df["BMXBMI"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Performance")
print("R² Score:", round(r2_score(y_test, predictions), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 3))
print("Root Mean Squared Error:", round(np.sqrt(mean_squared_error(y_test, predictions)), 3))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["Absolute"] = importance["Coefficient"].abs()
importance = importance.sort_values(by="Absolute", ascending=False)

print("\nFeature Importance")
print(importance[["Feature", "Coefficient"]])

plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("Actual vs Predicted BMI")
plt.show()

print("Project Complete!")
