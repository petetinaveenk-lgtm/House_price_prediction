# ==========================================
# HOUSE PRICE PREDICTION USING XGBOOST
# ==========================================

# Data Handling
import pandas as pd
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except Exception:
    # fall back to RandomForest if XGBoost is not installed
    from sklearn.ensemble import RandomForestRegressor
    XGBRegressor = None
    _USE_XGB = False

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

# Helper to find candidate files in the workspace
def find_file(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

train_candidates = [
    "train.csv",
    "./train.csv",
    r"C:/Users/navee/OneDrive/Desktop/CodeTech/train.csv",
]
test_candidates = [
    "test.csv",
    "./test.csv",
    r"C:/Users/navee/OneDrive/Desktop/CodeTech/test.csv",
]

train_path = find_file(train_candidates)
test_path = find_file(test_candidates)

def make_synthetic(train_p, test_p, n_train=200, n_test=100, seed=42):
    np.random.seed(seed)
    # basic numeric features needed by the pipeline
    def gen_df(n, include_target=True, start_id=1):
        Id = np.arange(start_id, start_id + n)
        TotalBsmtSF = np.random.randint(300, 2000, size=n)
        Flr1 = np.random.randint(400, 2000, size=n)
        Flr2 = np.random.randint(0, 1500, size=n)
        YearBuilt = np.random.randint(1900, 2010, size=n)
        YrSold = np.random.randint(2006, 2011, size=n)
        FullBath = np.random.randint(0, 4, size=n)
        HalfBath = np.random.randint(0, 2, size=n)
        BsmtFullBath = np.random.randint(0, 2, size=n)
        BsmtHalfBath = np.random.randint(0, 2, size=n)
        if include_target:
            SalePrice = np.random.randint(80000, 500000, size=n)
            df = pd.DataFrame({
                "Id": Id,
                "TotalBsmtSF": TotalBsmtSF,
                "1stFlrSF": Flr1,
                "2ndFlrSF": Flr2,
                "YearBuilt": YearBuilt,
                "YrSold": YrSold,
                "FullBath": FullBath,
                "HalfBath": HalfBath,
                "BsmtFullBath": BsmtFullBath,
                "BsmtHalfBath": BsmtHalfBath,
                "SalePrice": SalePrice,
            })
        else:
            df = pd.DataFrame({
                "Id": Id,
                "TotalBsmtSF": TotalBsmtSF,
                "1stFlrSF": Flr1,
                "2ndFlrSF": Flr2,
                "YearBuilt": YearBuilt,
                "YrSold": YrSold,
                "FullBath": FullBath,
                "HalfBath": HalfBath,
                "BsmtFullBath": BsmtFullBath,
                "BsmtHalfBath": BsmtHalfBath,
            })
        return df

    tr = gen_df(n_train, include_target=True, start_id=1)
    te = gen_df(n_test, include_target=False, start_id=1 + n_train)
    tr.to_csv(train_p, index=False)
    te.to_csv(test_p, index=False)
    return tr, te

if train_path is None or test_path is None:
    print("train/test files not found — creating small synthetic dataset for demo.")
    train_path = train_candidates[0]
    test_path = test_candidates[0]
    train_df, test_df = make_synthetic(train_path, test_path)
else:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

# Save IDs
test_ids = test_df["Id"]

# ==========================================
# FEATURE ENGINEERING
# ==========================================

for df in [train_df, test_df]:

    df["TotalSF"] = (
        df["TotalBsmtSF"] +
        df["1stFlrSF"] +
        df["2ndFlrSF"]
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    df["TotalBath"] = (
        df["FullBath"] +
        0.5 * df["HalfBath"] +
        df["BsmtFullBath"] +
        0.5 * df["BsmtHalfBath"]
    )

# ==========================================
# TARGET VARIABLE
# ==========================================

y = np.log1p(train_df["SalePrice"])

# Remove target from training data
train_df.drop("SalePrice", axis=1, inplace=True)

# ==========================================
# COMBINE TRAIN + TEST
# ==========================================

combined = pd.concat([train_df, test_df], axis=0)

# ==========================================
# HANDLE CATEGORICAL FEATURES
# ==========================================

combined = pd.get_dummies(combined)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

imputer = SimpleImputer(strategy="median")
combined = pd.DataFrame(
    imputer.fit_transform(combined),
    columns=combined.columns
)

# ==========================================
# SPLIT BACK
# ==========================================

X_train = combined[:len(train_df)]
X_test = combined[len(train_df):]

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

print("Training Model...")
model.fit(X_tr, y_tr)

# ==========================================
# VALIDATION
# ==========================================

pred_log = model.predict(X_val)

actual = np.expm1(y_val)
predictions = np.expm1(pred_log)

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("\nMODEL PERFORMANCE")
print("-" * 40)
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features")
print(importance.head(15))

# ==========================================
# RETRAIN ON FULL DATA
# ==========================================

model.fit(X_train, y)

# ==========================================
# TEST PREDICTIONS
# ==========================================

test_pred_log = model.predict(X_test)
test_predictions = np.expm1(test_pred_log)

# ==========================================
# SUBMISSION FILE
# ==========================================

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 5))
sns.histplot(test_predictions, bins=40)
plt.title("Predicted House Prices")
plt.show()

# ==========================================
# TOP FEATURES PLOT
# ==========================================

top_features = importance.head(15)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=top_features
)

plt.title("Top 15 Feature Importance")
plt.tight_layout()
plt.show()

print("\nProject Completed Successfully!")