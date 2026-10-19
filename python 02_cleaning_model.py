import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. LOAD RAW DATA
# ==========================================

df = pd.read_csv("raw_books.csv")

print("Raw dataset shape:", df.shape)

# ==========================================
# 2. DATA CLEANING
# ==========================================

# Remove duplicate rows
df = df.drop_duplicates()

# Clean Price and convert to numeric
df["Price"] = (
    df["Price"]
    .astype(str)
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
    .astype(float)
)

# Extract numeric stock quantity from Availability
df["Stock"] = (
    df["Availability"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
)

# Convert Stock to numeric
df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce")

# Fill missing stock values with 0
df["Stock"] = df["Stock"].fillna(0).astype(int)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

# Length of book title
df["Title_Length"] = df["Title"].astype(str).str.len()

# Price category
df["Price_Category"] = pd.cut(
    df["Price"],
    bins=[0, 20, 40, 60, 100],
    labels=["Low", "Medium", "High", "Premium"],
    include_lowest=True
)

# Rating category
df["Rating_Category"] = pd.cut(
    df["Rating"],
    bins=[0, 2, 3, 4, 5],
    labels=["Poor", "Average", "Good", "Excellent"],
    include_lowest=True
)

# ==========================================
# 4. CHECK DATA QUALITY
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nCleaned dataset shape:", df.shape)

# ==========================================
# 5. K-MEANS CLUSTERING
# ==========================================

# Features used for clustering
features = [
    "Price",
    "Rating",
    "Stock",
    "Title_Length"
]

X = df[features].copy()

# Make sure there are no missing values
X = X.fillna(X.median())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Fit the model
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ==========================================
# 6. CLUSTER ANALYSIS
# ==========================================

cluster_summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\n==========================================")
print("CLUSTER SUMMARY")
print("==========================================")
print(cluster_summary)

# ==========================================
# 7. SAVE CLEAN DATA
# ==========================================

df.to_csv("clean_books.csv", index=False)

print("\nCleaned dataset saved as: clean_books.csv")

# ==========================================
# 8. BASIC PROJECT INSIGHTS
# ==========================================

print("\n==========================================")
print("PROJECT INSIGHTS")
print("==========================================")

print("\nTotal books:", len(df))

print(
    "Average price: £",
    round(df["Price"].mean(), 2)
)

print(
    "Average rating:",
    round(df["Rating"].mean(), 2)
)

print(
    "Average stock:",
    round(df["Stock"].mean(), 2)
)

# ==========================================
# 9. BOOKS BY RATING
# ==========================================

print("\nBooks by rating:")
print(
    df["Rating"]
    .value_counts()
    .sort_index()
)

# ==========================================
# 10. BOOKS BY PRICE CATEGORY
# ==========================================

print("\nBooks by price category:")
print(
    df["Price_Category"]
    .value_counts()
)

# ==========================================
# 11. BOOKS BY CLUSTER
# ==========================================

print("\nBooks by cluster:")
print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

# ==========================================
# 12. MOST EXPENSIVE BOOKS
# ==========================================

print("\nTop 10 most expensive books:")

print(
    df[["Title", "Price", "Rating", "Cluster"]]
    .sort_values("Price", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ==========================================
# 13. HIGHEST RATED BOOKS
# ==========================================

print("\nTop 10 highest-rated books:")

print(
    df[["Title", "Price", "Rating", "Cluster"]]
    .sort_values(
        ["Rating", "Price"],
        ascending=[False, True]
    )
    .head(10)
    .to_string(index=False)
)

# ==========================================
# 14. FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("PROJECT DATA PROCESSING COMPLETED!")
print("==========================================")
print("Raw records:", len(pd.read_csv("raw_books.csv")))
print("Clean records:", len(df))
print("Output file: clean_books.csv")
print("Number of clusters:", df["Cluster"].nunique())
print("==========================================")