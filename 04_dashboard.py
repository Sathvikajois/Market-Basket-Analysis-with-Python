import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Book Market Analytics",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("clean_books.csv")

df = load_data()

# ==========================================
# TITLE
# ==========================================

st.title("📚 Book Market Analytics Dashboard")

st.markdown(
    """
    Interactive analysis of **1,000 books** collected through
    web scraping from Books to Scrape.
    
    The dashboard explores book pricing, ratings, inventory
    and machine-learning-based customer/product segments.
    """
)

st.divider()

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🔎 Dashboard Filters")

rating_options = sorted(df["Rating"].unique())

selected_ratings = st.sidebar.multiselect(
    "Select Rating",
    options=rating_options,
    default=rating_options
)

cluster_options = sorted(df["Cluster"].unique())

selected_clusters = st.sidebar.multiselect(
    "Select Cluster",
    options=cluster_options,
    default=cluster_options
)

min_price = float(df["Price"].min())
max_price = float(df["Price"].max())

price_range = st.sidebar.slider(
    "Select Price Range (£)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# ==========================================
# FILTER DATA
# ==========================================

filtered_df = df[
    (df["Rating"].isin(selected_ratings)) &
    (df["Cluster"].isin(selected_clusters)) &
    (df["Price"] >= price_range[0]) &
    (df["Price"] <= price_range[1])
]

# ==========================================
# KPI CARDS
# ==========================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📚 Total Books",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "💰 Average Price",
        f"£{filtered_df['Price'].mean():.2f}"
        if len(filtered_df) else "N/A"
    )

with col3:
    st.metric(
        "⭐ Average Rating",
        f"{filtered_df['Rating'].mean():.2f}/5"
        if len(filtered_df) else "N/A"
    )

with col4:
    st.metric(
        "📦 Average Stock",
        f"{filtered_df['Stock'].mean():.1f}"
        if len(filtered_df) else "N/A"
    )

st.divider()

# ==========================================
# PRICE DISTRIBUTION
# ==========================================

st.subheader("💰 Book Price Analysis")

fig_price = px.histogram(
    filtered_df,
    x="Price",
    nbins=20,
    title="Distribution of Book Prices",
    labels={"Price": "Book Price (£)"}
)

st.plotly_chart(
    fig_price,
    use_container_width=True
)

# ==========================================
# RATING DISTRIBUTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    rating_count = (
        filtered_df["Rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_count.columns = [
        "Rating",
        "Number of Books"
    ]

    fig_rating = px.bar(
        rating_count,
        x="Rating",
        y="Number of Books",
        title="Books by Rating"
    )

    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )

# ==========================================
# PRICE CATEGORY
# ==========================================

with col2:

    price_category = (
        filtered_df["Price_Category"]
        .value_counts()
        .reset_index()
    )

    price_category.columns = [
        "Price Category",
        "Number of Books"
    ]

    fig_category = px.pie(
        price_category,
        names="Price Category",
        values="Number of Books",
        title="Books by Price Category"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

st.divider()

# ==========================================
# PRICE VS RATING
# ==========================================

st.subheader("⭐ Price vs Rating Analysis")

fig_scatter = px.scatter(
    filtered_df,
    x="Rating",
    y="Price",
    color="Cluster",
    hover_name="Title",
    title="Price vs Rating by K-Means Cluster",
    labels={
        "Price": "Price (£)",
        "Rating": "Book Rating"
    }
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

# ==========================================
# MACHINE LEARNING SEGMENTS
# ==========================================

st.divider()

st.subheader("🤖 Machine Learning — Book Segmentation")

st.write(
    """
    K-Means clustering was used to divide books into three
    groups based on **price, rating, stock and title length**.
    """
)

cluster_summary = (
    filtered_df
    .groupby("Cluster")[
        ["Price", "Rating", "Stock", "Title_Length"]
    ]
    .mean()
    .round(2)
    .reset_index()
)

st.dataframe(
    cluster_summary,
    use_container_width=True
)

cluster_count = (
    filtered_df["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "Number of Books"
]

fig_cluster = px.bar(
    cluster_count,
    x="Cluster",
    y="Number of Books",
    title="Number of Books in Each ML Segment"
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# ==========================================
# BUSINESS INSIGHTS
# ==========================================

st.divider()

st.subheader("💡 Business Insights")

st.markdown(
    """
    **1. Pricing Strategy**
    
    The price distribution helps identify the dominant price
    ranges in the online book catalogue.

    **2. Rating Analysis**
    
    Comparing ratings with prices helps determine whether
    higher-priced books necessarily receive better ratings.

    **3. Inventory Management**
    
    Stock information can help identify inventory patterns
    and support catalogue management decisions.

    **4. Book Segmentation**
    
    K-Means clustering creates data-driven book segments
    based on pricing, ratings, stock and title characteristics.

    **5. Decision Support**
    
    The interactive filters allow decision-makers to examine
    specific price ranges, ratings and machine-learning segments.
    """
)

# ==========================================
# BOOK EXPLORER
# ==========================================

st.divider()

st.subheader("🔍 Book Explorer")

display_columns = [
    "Title",
    "Price",
    "Rating",
    "Stock",
    "Price_Category",
    "Rating_Category",
    "Cluster"
]

st.dataframe(
    filtered_df[display_columns]
    .sort_values("Rating", ascending=False),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# DOWNLOAD OPTION
# ==========================================

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="filtered_books.csv",
    mime="text/csv"
)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Capstone Project | Data Science | "
    "Web Scraping • EDA • Machine Learning • Dashboard"
)