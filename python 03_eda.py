import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD CLEAN DATA
# ==========================================

df = pd.read_csv("clean_books.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# ==========================================
# 2. BASIC STATISTICS
# ==========================================

print("\n==========================================")
print("BASIC STATISTICS")
print("==========================================")

print("\nTotal books:", len(df))
print("Average price: £", round(df["Price"].mean(), 2))
print("Average rating:", round(df["Rating"].mean(), 2))
print("Average stock:", round(df["Stock"].mean(), 2))

# ==========================================
# 3. RATING DISTRIBUTION
# ==========================================

print("\nRating distribution:")
print(df["Rating"].value_counts().sort_index())

plt.figure(figsize=(8, 5))

df["Rating"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("rating_distribution.png", dpi=300)
plt.show()

# ==========================================
# 4. PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Price"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
plt.tight_layout()

plt.savefig("price_distribution.png", dpi=300)
plt.show()

# ==========================================
# 5. PRICE VS RATING
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Rating"],
    df["Price"],
    alpha=0.6
)

plt.title("Book Price vs Rating")
plt.xlabel("Rating")
plt.ylabel("Price (£)")
plt.tight_layout()

plt.savefig("price_vs_rating.png", dpi=300)
plt.show()

# ==========================================
# 6. PRICE CATEGORY
# ==========================================

print("\nBooks by price category:")
print(df["Price_Category"].value_counts())

plt.figure(figsize=(8, 5))

df["Price_Category"].value_counts().plot(
    kind="bar"
)

plt.title("Books by Price Category")
plt.xlabel("Price Category")
plt.ylabel("Number of Books")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("price_category.png", dpi=300)
plt.show()

# ==========================================
# 7. CLUSTER DISTRIBUTION
# ==========================================

print("\nBooks by cluster:")
print(df["Cluster"].value_counts().sort_index())

plt.figure(figsize=(8, 5))

df["Cluster"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Book Segments Created by K-Means")
plt.xlabel("Cluster")
plt.ylabel("Number of Books")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("cluster_distribution.png", dpi=300)
plt.show()

# ==========================================
# 8. AVERAGE PRICE BY CLUSTER
# ==========================================

cluster_price = df.groupby("Cluster")["Price"].mean()

print("\nAverage price by cluster:")
print(cluster_price.round(2))

plt.figure(figsize=(8, 5))

cluster_price.plot(kind="bar")

plt.title("Average Book Price by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Price (£)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("cluster_price.png", dpi=300)
plt.show()

# ==========================================
# 9. AVERAGE RATING BY CLUSTER
# ==========================================

cluster_rating = df.groupby("Cluster")["Rating"].mean()

print("\nAverage rating by cluster:")
print(cluster_rating.round(2))

plt.figure(figsize=(8, 5))

cluster_rating.plot(kind="bar")

plt.title("Average Book Rating by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("cluster_rating.png", dpi=300)
plt.show()

# ==========================================
# 10. CORRELATION ANALYSIS
# ==========================================

print("\n==========================================")
print("CORRELATION MATRIX")
print("==========================================")

correlation = df[
    ["Price", "Rating", "Stock", "Title_Length"]
].corr()

print(correlation.round(2))

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

# ==========================================
# 11. TOP 10 MOST EXPENSIVE BOOKS
# ==========================================

print("\n==========================================")
print("TOP 10 MOST EXPENSIVE BOOKS")
print("==========================================")

print(
    df[
        ["Title", "Price", "Rating"]
    ]
    .sort_values("Price", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ==========================================
# 12. TOP 10 HIGHEST-RATED BOOKS
# ==========================================

print("\n==========================================")
print("TOP 10 HIGHEST-RATED BOOKS")
print("==========================================")

print(
    df[
        ["Title", "Price", "Rating"]
    ]
    .sort_values(
        ["Rating", "Price"],
        ascending=[False, True]
    )
    .head(10)
    .to_string(index=False)
)

# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("==========================================")
print("Charts saved in the project folder.")