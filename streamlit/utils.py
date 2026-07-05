from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RAW_DATASET_PATH = DATASET_DIR / "online_retail.csv"
CLEAN_DATASET_PATH = DATASET_DIR / "cleaned_online_retail.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
KMEANS_PATH = MODELS_DIR / "kmeans.pkl"
SIMILARITY_PATH = MODELS_DIR / "similarity.pkl"
PRODUCT_NAMES_PATH = MODELS_DIR / "product_names.pkl"
STYLES_PATH = BASE_DIR / "styles.css"

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
REQUIRED_RETAIL_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
DEFAULT_SEGMENT_NAMES = ["High Value", "Regular", "Occasional", "At-Risk"]
COUNTRY_ALIASES = {
    "EIRE": "Ireland",
    "USA": "United States",
    "RSA": "South Africa",
}


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    orders: int
    customers: int
    products: int
    countries: int
    revenue: float
    min_date: pd.Timestamp | None
    max_date: pd.Timestamp | None


@dataclass(frozen=True)
class SegmentModelBundle:
    scaler: StandardScaler
    kmeans: KMeans
    label_map: dict[int, str]
    ordered_clusters: list[int]
    cluster_centers: pd.DataFrame


@dataclass(frozen=True)
class RecommendationStore:
    matrix: pd.DataFrame | np.ndarray
    product_names: list[str]
    index_lookup: dict[str, int]
    source_type: str


def path_mtime_ns(path: Path) -> int:
    """Return a stable modification signature for cache keys."""

    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


def read_stylesheet() -> str:
    """Load the shared CSS bundle if it exists."""

    if not STYLES_PATH.exists():
        return ""
    return STYLES_PATH.read_text(encoding="utf-8")


def inject_styles() -> None:
    """Inject global CSS into Streamlit."""

    css = read_stylesheet()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def ensure_directory(path: Path) -> None:
    """Create a parent directory if it does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {', '.join(missing)}"
        )


def _coerce_retail_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the retail dataframe and remove rows that break the analysis."""

    if df is None or df.empty:
        raise ValueError("The retail dataset is empty.")

    working = df.copy()
    working.columns = [str(column).strip() for column in working.columns]

    _ensure_columns(working, REQUIRED_RETAIL_COLUMNS, "Retail dataset")

    numeric_columns = ["Quantity", "UnitPrice", "CustomerID"]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["InvoiceDate"] = pd.to_datetime(working["InvoiceDate"], errors="coerce")
    working["InvoiceNo"] = working["InvoiceNo"].astype("string").str.strip()
    working["StockCode"] = working["StockCode"].astype("string").str.strip()
    working["Description"] = working["Description"].astype("string").str.strip()
    working["Country"] = working["Country"].astype("string").str.strip()

    mask = (
        working["CustomerID"].notna()
        & working["Description"].notna()
        & working["Description"].astype(str).str.strip().ne("")
        & working["InvoiceDate"].notna()
        & working["Quantity"].gt(0)
        & working["UnitPrice"].gt(0)
        & ~working["InvoiceNo"].astype(str).str.startswith("C", na=False)
    )
    working = working.loc[mask].copy()

    if working.empty:
        raise ValueError("No valid retail rows remain after cleaning.")

    working["CustomerID"] = working["CustomerID"].round().astype("Int64")
    working["TotalAmount"] = (working["Quantity"] * working["UnitPrice"]).astype(
        "float64"
    )
    working["InvoiceNo"] = working["InvoiceNo"].astype("string").str.strip()
    working["StockCode"] = working["StockCode"].astype("string").str.strip()
    working["Description"] = working["Description"].astype("string").str.strip()
    working["Country"] = working["Country"].astype("string").str.strip()

    working = working.sort_values(
        by=["InvoiceDate", "InvoiceNo", "StockCode"],
        kind="mergesort",
    ).reset_index(drop=True)
    return working


def ensure_clean_dataset(refresh: bool = False) -> Path:
    """
    Ensure the cleaned retail dataset exists.

    The function regenerates the cleaned CSV when the raw dataset is newer,
    the cleaned file is missing, or refresh=True.
    """

    if not RAW_DATASET_PATH.exists() and CLEAN_DATASET_PATH.exists() and not refresh:
        return CLEAN_DATASET_PATH

    if (
        CLEAN_DATASET_PATH.exists()
        and not refresh
        and (
            not RAW_DATASET_PATH.exists()
            or path_mtime_ns(CLEAN_DATASET_PATH) >= path_mtime_ns(RAW_DATASET_PATH)
        )
    ):
        try:
            preview = pd.read_csv(CLEAN_DATASET_PATH, nrows=5, low_memory=False)
            _ensure_columns(preview, REQUIRED_RETAIL_COLUMNS + ["TotalAmount"], "Cleaned dataset")
            return CLEAN_DATASET_PATH
        except Exception:
            pass

    if not RAW_DATASET_PATH.exists():
        if CLEAN_DATASET_PATH.exists():
            return CLEAN_DATASET_PATH
        raise FileNotFoundError(
            "Neither dataset/online_retail.csv nor dataset/cleaned_online_retail.csv is available."
        )

    raw_df = pd.read_csv(RAW_DATASET_PATH, low_memory=False)
    clean_df = _coerce_retail_frame(raw_df)
    ensure_directory(CLEAN_DATASET_PATH)
    clean_df.to_csv(
        CLEAN_DATASET_PATH,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    return CLEAN_DATASET_PATH


@st.cache_data(show_spinner=False)
def _load_clean_dataset_cached(clean_path: str, clean_mtime_ns: int) -> pd.DataFrame:
    """Cached loader for the cleaned dataset."""

    df = pd.read_csv(clean_path, low_memory=False)
    return _coerce_retail_frame(df)


def load_clean_dataset(refresh: bool = False) -> pd.DataFrame:
    """Load the cleaned retail dataset and normalize it for the app."""

    clean_path = ensure_clean_dataset(refresh=refresh)
    return _load_clean_dataset_cached(str(clean_path), path_mtime_ns(clean_path))


def dataset_summary(df: pd.DataFrame) -> DatasetSummary:
    """Create a quick summary for sidebar cards and page headers."""

    if df.empty:
        return DatasetSummary(0, 0, 0, 0, 0, 0.0, None, None)

    _ensure_columns(df, ["InvoiceNo", "Description", "CustomerID", "Country", "InvoiceDate", "TotalAmount"], "Dataset summary")

    return DatasetSummary(
        rows=int(len(df)),
        orders=int(df["InvoiceNo"].nunique()),
        customers=int(df["CustomerID"].nunique()),
        products=int(df["Description"].nunique()),
        countries=int(df["Country"].nunique()),
        revenue=float(df["TotalAmount"].sum()),
        min_date=df["InvoiceDate"].min(),
        max_date=df["InvoiceDate"].max(),
    )


def format_currency(value: float | int | None, symbol: str = "₹") -> str:
    """Format a currency value for display."""

    if pd.isna(value):
        return f"{symbol}0.00"
    return f"{symbol}{float(value):,.2f}"


def format_compact_currency(value: float | int | None, symbol: str = "₹") -> str:
    """Format a currency value with compact K/M/B units for KPI cards."""

    if pd.isna(value):
        return f"{symbol}0.00"

    amount = float(value)
    sign = "-" if amount < 0 else ""
    absolute_amount = abs(amount)

    suffixes = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for index, (threshold, suffix) in enumerate(suffixes):
        if absolute_amount >= threshold:
            compact_value = absolute_amount / threshold
            if round(compact_value, 2) >= 1000 and index > 0:
                higher_threshold, higher_suffix = suffixes[index - 1]
                compact_value = absolute_amount / higher_threshold
                return f"{sign}{symbol}{compact_value:.2f}{higher_suffix}"
            return f"{sign}{symbol}{compact_value:.2f}{suffix}"

    return f"{sign}{symbol}{absolute_amount:,.2f}"


def format_number(value: float | int | None, decimals: int = 0) -> str:
    """Format a numeric value with thousands separators."""

    if pd.isna(value):
        return "0"
    return f"{float(value):,.{decimals}f}" if decimals else f"{int(round(float(value))):,}"


def clean_text(value: object) -> str:
    """Return a trimmed string for display and lookup purposes."""

    if value is None:
        return "Unknown"

    try:
        if pd.isna(value):
            return "Unknown"
    except Exception:
        pass

    return str(value).strip()


def display_text(value: object) -> str:
    """Normalize user-facing text without mutating source data."""

    return clean_text(value).replace("£", "₹")


def format_customer_id(customer_id: object) -> str:
    """Display customer IDs consistently."""

    if pd.isna(customer_id):
        return "Unknown"
    try:
        return str(int(float(customer_id)))
    except Exception:
        return str(customer_id)


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | list | None = None,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """Filter the retail dataset by date range and country selection."""

    if df.empty:
        return df.copy()

    working = df.copy()

    if date_range and len(date_range) == 2:
        start_raw, end_raw = date_range
        start = pd.Timestamp(start_raw).normalize()
        end = pd.Timestamp(end_raw).normalize() + pd.Timedelta(days=1)
        working = working.loc[
            (working["InvoiceDate"] >= start) & (working["InvoiceDate"] < end)
        ]

    if countries:
        working = working.loc[working["Country"].isin(countries)]

    return working.reset_index(drop=True)


def get_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Return the core dashboard KPIs."""

    if df.empty:
        return {
            "revenue": 0.0,
            "customers": 0.0,
            "products": 0.0,
            "orders": 0.0,
            "line_items": 0.0,
            "aov": 0.0,
        }

    _ensure_columns(df, ["InvoiceNo", "Description", "CustomerID", "TotalAmount"], "KPI calculation")

    orders = float(df["InvoiceNo"].nunique())
    revenue = float(df["TotalAmount"].sum())
    customers = float(df["CustomerID"].nunique())
    products = float(df["Description"].nunique())
    line_items = float(len(df))
    average_order_value = revenue / orders if orders else 0.0

    return {
        "revenue": revenue,
        "customers": customers,
        "products": products,
        "orders": orders,
        "line_items": line_items,
        "aov": average_order_value,
    }


def get_top_products(
    df: pd.DataFrame,
    metric: str = "Revenue",
    top_n: int = 10,
) -> pd.DataFrame:
    """Aggregate products by revenue, quantity, and orders."""

    _ensure_columns(df, ["Description", "Quantity", "TotalAmount", "InvoiceNo"], "Top products")

    if metric not in {"Revenue", "Quantity", "Orders"}:
        raise ValueError(
            "Top products metric must be one of: Revenue, Quantity, Orders."
        )

    summary = (
        df.groupby("Description", dropna=True)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index()
    )
    return summary


def get_top_customers(
    df: pd.DataFrame,
    metric: str = "Revenue",
    top_n: int = 10,
) -> pd.DataFrame:
    """Aggregate customers by revenue or order count."""

    _ensure_columns(df, ["CustomerID", "InvoiceNo", "Quantity", "TotalAmount"], "Top customers")

    if metric not in {"Revenue", "Orders", "Quantity"}:
        raise ValueError(
            "Top customers metric must be one of: Revenue, Orders, Quantity."
        )

    summary = (
        df.groupby("CustomerID", dropna=True)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Quantity=("Quantity", "sum"),
            LastPurchase=("InvoiceDate", "max"),
        )
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index()
    )
    summary["AverageOrderValue"] = summary["Revenue"] / summary["Orders"].replace(0, np.nan)
    return summary


def get_country_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Aggregate customer activity by country."""

    _ensure_columns(df, ["Country", "InvoiceNo", "CustomerID", "Quantity", "TotalAmount"], "Country analysis")

    summary = (
        df.groupby("Country", dropna=True)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
            Items=("Quantity", "sum"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    summary["CountryPlot"] = summary["Country"].replace(COUNTRY_ALIASES)
    summary["RevenueShare"] = summary["Revenue"] / summary["Revenue"].sum()
    return summary.head(top_n)


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly sales using period-based grouping for pandas version safety."""

    _ensure_columns(df, ["InvoiceDate", "InvoiceNo", "CustomerID", "Quantity", "TotalAmount"], "Monthly sales")

    monthly = (
        df.assign(Month=df["InvoiceDate"].dt.to_period("M"))
        .groupby("Month", dropna=True)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
            Items=("Quantity", "sum"),
        )
        .reset_index()
    )
    monthly["MonthLabel"] = monthly["Month"].astype(str)
    return monthly


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the RFM table used for customer segmentation."""

    _ensure_columns(df, ["InvoiceDate", "InvoiceNo", "CustomerID", "TotalAmount"], "RFM analysis")

    if df.empty:
        raise ValueError("RFM analysis requires at least one transaction.")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda values: int((snapshot_date - values.max()).days)),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .sort_values(["Monetary", "Frequency", "Recency"], ascending=[False, False, True])
    )
    rfm["Recency"] = rfm["Recency"].astype("int64")
    rfm["Frequency"] = rfm["Frequency"].astype("int64")
    rfm["Monetary"] = rfm["Monetary"].astype("float64")
    rfm.index.name = "CustomerID"
    return rfm


@st.cache_resource(show_spinner=False)
def _load_joblib_cached(artifact_path: str, artifact_mtime_ns: int):
    """Cached joblib loader keyed by modification time."""

    return joblib.load(artifact_path)


def load_joblib_artifact(path: Path, artifact_name: str):
    """Load a serialized artifact with friendly errors."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {artifact_name} artifact: {path}")

    try:
        return _load_joblib_cached(str(path), path_mtime_ns(path))
    except Exception as exc:  # pragma: no cover - defensive user-facing error
        raise RuntimeError(f"Unable to load {artifact_name} from {path.name}: {exc}") from exc


def validate_segmentation_models(
    df: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
) -> bool:
    """Check whether the saved segmentation artifacts can work with the dataset."""

    if df.empty:
        return False

    if not hasattr(scaler, "transform") or not hasattr(kmeans, "predict"):
        return False

    try:
        rfm = build_rfm_table(df)
        scaled = scaler.transform(rfm[RFM_COLUMNS])
        if scaled.shape[1] != len(RFM_COLUMNS):
            return False
        sample_count = min(len(rfm), 10)
        if sample_count <= 0:
            return False
        predictions = kmeans.predict(scaled[:sample_count])
        return predictions.shape[0] == sample_count
    except Exception:
        return False


def train_segmentation_models(df: pd.DataFrame) -> tuple[StandardScaler, KMeans]:
    """Train fresh segmentation artifacts and persist them to disk."""

    rfm = build_rfm_table(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[RFM_COLUMNS])

    n_clusters = min(4, len(rfm))
    if n_clusters < 2:
        raise ValueError("Customer segmentation requires at least two customers.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled)

    ensure_directory(SCALER_PATH)
    ensure_directory(KMEANS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    return scaler, kmeans


def build_segment_label_map(
    scaler: StandardScaler,
    kmeans: KMeans,
) -> tuple[dict[int, str], list[int], pd.DataFrame]:
    """Create stable customer segment labels from model cluster centers."""

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=RFM_COLUMNS,
    )
    centers["Cluster"] = range(len(centers))
    centers = centers[["Cluster", "Recency", "Frequency", "Monetary"]]

    ordered_clusters = (
        centers.sort_values(
            by=["Monetary", "Frequency", "Recency"],
            ascending=[False, False, True],
        )["Cluster"]
        .astype(int)
        .tolist()
    )

    if len(ordered_clusters) == 4:
        labels = DEFAULT_SEGMENT_NAMES
    else:
        labels = [f"Segment {index + 1}" for index in range(len(ordered_clusters))]

    label_map = {
        cluster: labels[position]
        for position, cluster in enumerate(ordered_clusters)
    }
    centers["Label"] = centers["Cluster"].map(label_map)
    return label_map, ordered_clusters, centers


def load_segmentation_bundle(df: pd.DataFrame) -> SegmentModelBundle:
    """Load saved segmentation artifacts, retraining only when required."""

    scaler: StandardScaler | None = None
    kmeans: KMeans | None = None

    if SCALER_PATH.exists():
        try:
            scaler = load_joblib_artifact(SCALER_PATH, "scaler")
        except Exception:
            scaler = None

    if KMEANS_PATH.exists():
        try:
            kmeans = load_joblib_artifact(KMEANS_PATH, "kmeans")
        except Exception:
            kmeans = None

    if scaler is None or kmeans is None or not validate_segmentation_models(df, scaler, kmeans):
        scaler, kmeans = train_segmentation_models(df)

    label_map, ordered_clusters, cluster_centers = build_segment_label_map(scaler, kmeans)
    return SegmentModelBundle(
        scaler=scaler,
        kmeans=kmeans,
        label_map=label_map,
        ordered_clusters=ordered_clusters,
        cluster_centers=cluster_centers,
    )


def assign_customer_segments(
    rfm: pd.DataFrame,
    bundle: SegmentModelBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach cluster and segment labels to an RFM table."""

    if rfm.empty:
        raise ValueError("Customer segmentation requires at least one customer.")

    _ensure_columns(rfm, RFM_COLUMNS, "Customer segmentation")

    scaled = bundle.scaler.transform(rfm[RFM_COLUMNS])
    predictions = bundle.kmeans.predict(scaled)

    segmented = rfm.copy()
    segmented["Cluster"] = predictions.astype(int)
    segmented["Segment"] = segmented["Cluster"].map(bundle.label_map)
    segmented["Segment"] = segmented["Segment"].fillna(
        segmented["Cluster"].map(lambda cluster: f"Cluster {cluster}")
    )

    summary = (
        segmented.groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .reset_index()
    )
    summary["Segment"] = summary["Cluster"].map(bundle.label_map)
    summary["Share"] = summary["Customers"] / summary["Customers"].sum()
    summary["RevenueShare"] = (
        segmented.groupby("Cluster")["Monetary"].sum().reindex(summary["Cluster"]).values
        / segmented["Monetary"].sum()
    )

    ordered_clusters = [cluster for cluster in bundle.ordered_clusters if cluster in set(summary["Cluster"])]
    if ordered_clusters:
        summary["SortOrder"] = summary["Cluster"].map({cluster: index for index, cluster in enumerate(ordered_clusters)})
        summary = summary.sort_values("SortOrder").drop(columns="SortOrder")

    segmented = segmented.sort_values(["Monetary", "Frequency", "Recency"], ascending=[False, False, True])
    return segmented, summary


def normalize_similarity_object(
    similarity_obj,
    product_names_obj: list[str] | None = None,
) -> RecommendationStore:
    """Normalize the saved recommendation artifact into a common store."""

    if isinstance(similarity_obj, pd.DataFrame):
        matrix = similarity_obj.copy()
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        if not matrix.index.equals(matrix.columns):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("similarity.pkl must be a square similarity matrix.")
            matrix.columns = matrix.index

        product_names = matrix.index.astype(str).tolist()
        index_lookup = {name: position for position, name in enumerate(product_names)}
        return RecommendationStore(
            matrix=matrix,
            product_names=product_names,
            index_lookup=index_lookup,
            source_type="DataFrame",
        )

    if isinstance(similarity_obj, np.ndarray):
        if product_names_obj is None:
            raise ValueError(
                "similarity.pkl is a NumPy array, so product_names.pkl is required."
            )
        matrix = np.asarray(similarity_obj, dtype=float)
        product_names = [str(name) for name in product_names_obj]
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("similarity.pkl must contain a square matrix.")
        if matrix.shape[0] != len(product_names):
            raise ValueError(
                "The similarity matrix and product_names.pkl have incompatible sizes."
            )
        index_lookup = {name: position for position, name in enumerate(product_names)}
        return RecommendationStore(
            matrix=matrix,
            product_names=product_names,
            index_lookup=index_lookup,
            source_type="NumPy array",
        )

    raise TypeError(
        f"Unsupported similarity artifact type: {type(similarity_obj).__name__}"
    )


def load_recommendation_store():
    """Load recommendation model, downloading it from Google Drive if needed."""

    if not SIMILARITY_PATH.exists():
        SIMILARITY_PATH.parent.mkdir(parents=True, exist_ok=True)

        file_id = "1HXlJxRNt6P7ucwiGZWX-tZx29dVsg7Ss"

        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(SIMILARITY_PATH),
            quiet=False,
        )

    similarity_obj = load_joblib_artifact(SIMILARITY_PATH, "similarity")

    product_names_obj = None

    if PRODUCT_NAMES_PATH.exists():
        product_names_obj = load_joblib_artifact(
            PRODUCT_NAMES_PATH,
            "product names",
        )

    return normalize_similarity_object(
        similarity_obj,
        product_names_obj,
    )
def search_products(product_names: list[str], query: str, limit: int = 100) -> list[str]:
    """Return product matches for a text query."""

    cleaned_query = display_text(query).lower()
    if not cleaned_query:
        return product_names[:limit]

    matches = [
        product_name
        for product_name in product_names
        if cleaned_query in display_text(product_name).lower()
    ]
    return matches[:limit]


def recommend_products(
    product_name: str,
    store: RecommendationStore,
    top_n: int = 5,
) -> pd.DataFrame:
    """Generate product recommendations with cosine similarity scores."""

    if not product_name:
        return pd.DataFrame(columns=["Rank", "Product", "Similarity"])

    if isinstance(store.matrix, pd.DataFrame):
        if product_name not in store.matrix.index:
            return pd.DataFrame(columns=["Rank", "Product", "Similarity"])

        scores = store.matrix.loc[product_name].sort_values(ascending=False)
        scores = scores.drop(labels=[product_name], errors="ignore").head(top_n)
        return pd.DataFrame(
            {
                "Rank": range(1, len(scores) + 1),
                "Product": scores.index.tolist(),
                "Similarity": scores.values.astype(float).tolist(),
            }
        )

    index = store.index_lookup.get(product_name)
    if index is None:
        return pd.DataFrame(columns=["Rank", "Product", "Similarity"])

    scores = np.asarray(store.matrix[index], dtype=float)
    order = np.argsort(scores)[::-1]
    recommendations: list[dict[str, object]] = []

    for candidate_index in order:
        if candidate_index == index:
            continue
        recommendations.append(
            {
                "Rank": len(recommendations) + 1,
                "Product": store.product_names[candidate_index],
                "Similarity": float(scores[candidate_index]),
            }
        )
        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


def product_label(product_name: str) -> str:
    """Create a cleaner label for long product names."""

    return display_text(product_name)


def product_key(product_name: object) -> str:
    """Return the raw product name used for comparisons and lookups."""

    return clean_text(product_name)
