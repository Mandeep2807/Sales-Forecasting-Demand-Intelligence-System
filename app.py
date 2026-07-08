import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Demand Intelligence Dashboard")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    data["Order Date"] = pd.to_datetime(
    data["Order Date"],
    format="%d/%m/%Y"
)
    return data

df = load_data()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# =====================================================
# PAGE 1 : SALES OVERVIEW
# =====================================================


if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    # ============================================
    # FILTERS
    # ============================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        region = st.selectbox(
            "🌍 Region",
            ["All"] + sorted(df["Region"].unique())
        )

    with c2:
        category = st.selectbox(
            "📦 Category",
            ["All"] + sorted(df["Category"].unique())
        )

    with c3:
        segment = st.selectbox(
            "👥 Segment",
            ["All"] + sorted(df["Segment"].unique())
        )

    with c4:
        years = sorted(df["Order Date"].dt.year.unique())

        year = st.selectbox(
            "📅 Year",
            ["All"] + years
        )

    # ============================================
    # APPLY FILTERS
    # ============================================

    filtered = df.copy()

    if region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]

    if category != "All":
        filtered = filtered[
            filtered["Category"] == category
        ]

    if segment != "All":
        filtered = filtered[
            filtered["Segment"] == segment
        ]

    if year != "All":
        filtered = filtered[
            filtered["Order Date"].dt.year == year
        ]

    if filtered.empty:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # ============================================
    # KPI CARDS
    # ============================================

    total_sales = filtered["Sales"].sum()
    total_orders = len(filtered)
    avg_sales = filtered["Sales"].mean()

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    k2.metric(
        "🛒 Orders",
        f"{total_orders:,}"
    )

    k3.metric(
        "📈 Average Sales",
        f"${avg_sales:,.2f}"
    )


    st.divider()

    # ============================================
    # CHART ROW 1
    # ============================================

    left, right = st.columns(2)

    with left:

        yearly = (
            filtered
            .groupby(filtered["Order Date"].dt.year)["Sales"]
            .sum()
        )

        fig, ax = plt.subplots(figsize=(6,4))

        yearly.plot(
            kind="bar",
            color="royalblue",
            ax=ax
        )

        ax.set_title("Yearly Sales")
        ax.set_ylabel("Sales")

        st.pyplot(fig)

    with right:

        monthly = (
            filtered
            .groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                )
            )["Sales"]
            .sum()
        )

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(
            monthly.index,
            monthly.values,
            linewidth=2.5,
            marker="o"
        )

        ax.set_title("Monthly Sales Trend")
        ax.set_ylabel("Sales")

        st.pyplot(fig)

    # ============================================
    # CHART ROW 2
    # ============================================

    left, right = st.columns(2)

    with left:

        region_sales = (
            filtered
            .groupby("Region")["Sales"]
            .sum()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(6,4))

        region_sales.plot(
            kind="barh",
            color="teal",
            ax=ax
        )

        ax.set_title("Sales by Region")

        st.pyplot(fig)

    with right:

        category_sales = (
            filtered
            .groupby("Category")["Sales"]
            .sum()
        )

        fig, ax = plt.subplots(figsize=(6,4))

        ax.pie(
            category_sales,
            labels=category_sales.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Category Share")

        st.pyplot(fig)

    # ============================================
    # TOP PRODUCTS
    # ============================================

    st.subheader("🏆 Top 10 Products")

    top_products = (
        filtered
        .groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,6))

    top_products.sort_values().plot(
        kind="barh",
        color="orange",
        ax=ax
    )

    ax.set_title("Top 10 Products by Sales")

    st.pyplot(fig)

    # ============================================
    # DATA TABLE
    # ============================================

    st.subheader("📄 Filtered Dataset")

    st.dataframe(
        filtered,
        use_container_width=True
    )

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Filtered Data",
        csv,
        "Filtered_Sales.csv",
        "text/csv"
    )
# =====================================================
# PAGE 2 : FORECAST
# =====================================================

elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    months = st.slider(
        "Forecast Months",
        1,
        3,
        3
    )

    def simple_forecast(series, steps):

        model = SARIMAX(
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)

        return result.forecast(steps)

    if forecast_type == "Category":

        selected = st.selectbox(
            "Category",
            sorted(df["Category"].unique())
        )

        sales = (
            df[df["Category"] == selected]
            .groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                )
            )["Sales"]
            .sum()
        )

    else:

        selected = st.selectbox(
            "Region",
            sorted(df["Region"].unique())
        )

        sales = (
            df[df["Region"] == selected]
            .groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                )
            )["Sales"]
            .sum()
        )

    if len(sales) < 12:
        st.warning("Not enough historical data available to generate a reliable forecast.")
        st.stop()
    
    prediction = simple_forecast(
        sales,
        months
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        sales.index,
        sales.values,
        label="Historical Sales",
        linewidth=2
    )

    future_dates = pd.date_range(
        start=sales.index[-1] + pd.offsets.MonthEnd(1),
        periods=months,
        freq="ME"
    )

    ax.plot(
        future_dates,
        prediction,
        marker="o",
        linewidth=2,
        label="Forecast"
    )

    ax.set_title("Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

# =====================================================
# PAGE 3 : ANOMALY REPORT
# =====================================================

elif page == "Anomaly Report":

    st.header("Sales Anomaly Detection")

    weekly = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly["Anomaly"] = iso.fit_predict(
        weekly[["Sales"]]
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly["Order Date"],
        weekly["Sales"]
    )

    outliers = weekly[
        weekly["Anomaly"] == -1
    ]

    ax.scatter(
        outliers["Order Date"],
        outliers["Sales"],
        color="red",
        s=70
    )

    ax.set_title("Detected Sales Anomalies")

    st.pyplot(fig)

    st.subheader("Anomaly Table")

    st.dataframe(outliers)

# =====================================================
# PAGE 4 : DEMAND SEGMENTS
# =====================================================

elif page == "Demand Segments":

    st.header("Product Demand Segments")

    cluster = (
        df.groupby("Sub-Category")["Sales"]
        .agg(["sum","mean","std"])
    )

    cluster.columns = [
        "Sales",
        "Average",
        "Volatility"
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(cluster)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    cluster["Cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)

    points = pca.fit_transform(scaled)

    plot = pd.DataFrame({
        "PC1": points[:,0],
        "PC2": points[:,1],
        "Cluster": cluster["Cluster"]
    })

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=plot,
        x="PC1",
        y="PC2",
        hue="Cluster",
        s=120,
        ax=ax
    )

    ax.set_title("Demand Segments")

    st.pyplot(fig)

    st.subheader("Cluster Details")

    st.dataframe(cluster)
