import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("OTS_Scheduling_Dataset.csv")

# Feature engineering
# Convert StartTime to datetime
df["StartTime"] = pd.to_datetime(df["StartTime"])

# Extract hour of operation start
df["StartHour"] = df["StartTime"].dt.hour

# Define features and target
X = df[["ProposedProcedure","DisplayValue","StartHour","TurnAroundTime"]]
y = df["OperationDuration"]

# Preprocessing
categorical_features = ["ProposedProcedure", "DisplayValue"]
numeric_features = ["StartHour", "TurnAroundTime"]

preprocessor = ColumnTransformer(
    transformers=[
        ("healthcare", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Build pipeline
model_v2 = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model_v2.fit(X, y)

joblib.dump(model_v2, "improved_model_v2.pkl")
print("Improved model saved.")