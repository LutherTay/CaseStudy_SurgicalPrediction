import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("OTS_Scheduling_Dataset.csv")

# Define features and target
X = df[["ProposedProcedure", "DisplayValue"]]
y = df["OperationDuration"]

# Preprocessing
categorical_features = ["ProposedProcedure", "DisplayValue"]

preprocessor = ColumnTransformer(
    transformers=[
        ("healthcare", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Build pipeline
model_v1 = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model_v1.fit(X, y)

joblib.dump(model_v1, "baseline_model_v1.pkl")
print("Baseline model saved.")