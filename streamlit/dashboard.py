from __future__ import annotations

import html
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils


PRIMARY_COLOR = "#0f766e"
SECONDARY_COLOR = "#1d4ed8"
ACCENT_COLOR = "#f97316"
MUTED_COLOR = "#64748b"
BACKGROUND_COLOR = "rgba(0, 0, 0, 0)"


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-kicker">Dashboard</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(title: str, value: str, caption: str, tone: str = "primary") -> str:
    return dedent(
        f"""
        <div class="metric-card metric-card--{tone}">
            <div class="metric-label">{html.escape(title)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
        </div>
        """
    ).strip()


def _render_metric_grid(summary: utils.DatasetSummary, kpis: dict[str, float]) -> None:
    cards = [
        _metric_card(
            "Revenue",
            utils.format_compact_currency(kpis["revenue"]),
            "Gross transaction value",
            "teal",
        ),
        _metric_card(
            "Customers",
            utils.format_number(kpis["customers"]),
            "Unique customers in view",
            "blue",
        ),
        _metric_card(
            "Products",
            utils.format_number(kpis["products"]),
            "Unique product descriptions",
            "amber",
        ),
        _metric_card(
            "Orders",
            utils.format_number(kpis["orders"]),
            "Unique invoices",
            "indigo",
        ),
        _metric_card(
            "Average order value",
            utils.format_currency(kpis["aov"]),
            "Revenue / orders",
            "slate",
        ),
    ]

    if len(cards) != 5:
        raise RuntimeError(f"Expected 5 dashboard KPI cards, got {len(cards)}.")

    st.markdown(
        dedent(
            f"""
        <div class="metric-grid">
            {''.join(cards)}
        </div>
        """,
        ).strip(),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Dataset window: {summary.min_date:%d %b %Y} to {summary.max_date:%d %b %Y}"
        if summary.min_date is not None and summary.max_date is not None
        else "Dataset window unavailable."
    )

    clean_dataset_path = utils.ensure_clean_dataset()
    st.download_button(
        "📥 Download Cleaned Dataset",
        data=clean_dataset_path.read_bytes(),
        file_name=clean_dataset_path.name,
        mime="text/csv",
        key="dashboard_cleaned_dataset_download",
    )


def _prepare_top_products(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    products = utils.get_top_products(df, metric=metric, top_n=top_n)
    display = products.copy()
    display["Description"] = display["Description"].map(utils.display_text)
    if metric == "Revenue":
        display["Revenue"] = display["Revenue"].map(utils.format_currency)
    if metric == "Quantity":
        display["Quantity"] = display["Quantity"].map(utils.format_number)
    display["Orders"] = display["Orders"].map(utils.format_number)
    return products, display


def _prepare_top_customers(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    customers = utils.get_top_customers(df, metric=metric, top_n=top_n)
    display = customers.copy()
    display["CustomerID"] = display["CustomerID"].map(utils.format_customer_id)
    display["Revenue"] = display["Revenue"].map(utils.format_currency)
    display["Orders"] = display["Orders"].map(utils.format_number)
    display["Quantity"] = display["Quantity"].map(utils.format_number)
    display["AverageOrderValue"] = display["AverageOrderValue"].map(utils.format_currency)
    display["LastPurchase"] = pd.to_datetime(display["LastPurchase"]).dt.strftime("%d %b %Y")
    return customers, display


def _build_monthly_chart(monthly: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=monthly["MonthLabel"],
            y=monthly["Revenue"],
            mode="lines+markers",
            line=dict(color=PRIMARY_COLOR, width=3),
            marker=dict(size=7, color=ACCENT_COLOR),
            name="Revenue",
            hovertemplate="%{x}<br>Revenue: ₹ %{y:,.2f}<extra></extra>",
        )
    )
    rolling = monthly["Revenue"].rolling(window=3, min_periods=1).mean()
    figure.add_trace(
        go.Scatter(
            x=monthly["MonthLabel"],
            y=rolling,
            mode="lines",
            line=dict(color=SECONDARY_COLOR, width=2, dash="dot"),
            name="3-month moving average",
            hovertemplate="%{x}<br>Average: ₹ %{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Monthly sales trend",
        template="plotly_white",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#0f172a"),
    )
    figure.update_xaxes(title_text="Month", showgrid=False)
    figure.update_yaxes(title_text="Revenue (₹)", gridcolor="#e2e8f0")
    return figure


def _build_bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str,
    value_format: str,
    currency_symbol: str | None = None,
) -> go.Figure:
    figure = px.bar(
        frame.sort_values(x, ascending=True),
        x=x,
        y=y,
        orientation="h",
        text=x,
        color_discrete_sequence=[color],
    )
    if currency_symbol:
        figure.update_traces(
            texttemplate=f"{currency_symbol}%{{x:,.2f}}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"{y}: %{{y}}<br>{x}: {currency_symbol}%{{x:,.2f}}<extra></extra>",
        )
    else:
        figure.update_traces(
            texttemplate=value_format,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"{y}: %{{y}}<br>{x}: %{{x:,.2f}}<extra></extra>",
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#0f172a"),
    )
    figure.update_xaxes(gridcolor="#e2e8f0")
    figure.update_yaxes(title_text="")
    return figure


def _build_country_choropleth(frame: pd.DataFrame) -> go.Figure:
    figure = px.choropleth(
        frame,
        locations="CountryPlot",
        locationmode="country names",
        color="Revenue",
        hover_name="Country",
        color_continuous_scale=["#cffafe", "#14b8a6", "#0f766e", "#134e4a"],
    )
    figure.update_geos(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
    )
    figure.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Revenue: ₹ %{z:,.2f}<extra></extra>"
    )
    figure.update_layout(
        title="Revenue by country",
        template="plotly_white",
        paper_bgcolor=BACKGROUND_COLOR,
        margin=dict(l=10, r=10, t=60, b=10),
        height=450,
        coloraxis_colorbar=dict(title="Revenue (₹)"),
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#0f172a"),
    )
    return figure


def _format_products_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    display = frame.copy()
    display["Description"] = display["Description"].map(utils.display_text)
    display["Revenue"] = display["Revenue"].map(utils.format_currency)
    display["Quantity"] = display["Quantity"].map(utils.format_number)
    display["Orders"] = display["Orders"].map(utils.format_number)
    columns = ["Description", "Revenue", "Quantity", "Orders"]
    if metric == "Quantity":
        columns = ["Description", "Quantity", "Revenue", "Orders"]
    return display[columns]


def _format_customers_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["CustomerID"] = display["CustomerID"].map(utils.format_customer_id)
    display["Revenue"] = display["Revenue"].map(utils.format_currency)
    display["Orders"] = display["Orders"].map(utils.format_number)
    display["Quantity"] = display["Quantity"].map(utils.format_number)
    display["AverageOrderValue"] = display["AverageOrderValue"].map(utils.format_currency)
    display["LastPurchase"] = pd.to_datetime(display["LastPurchase"]).dt.strftime("%d %b %Y")
    return display[
        ["CustomerID", "Revenue", "Orders", "Quantity", "AverageOrderValue", "LastPurchase"]
    ]


def _format_country_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["Revenue"] = display["Revenue"].map(utils.format_currency)
    display["Orders"] = display["Orders"].map(utils.format_number)
    display["Customers"] = display["Customers"].map(utils.format_number)
    display["Items"] = display["Items"].map(utils.format_number)
    display["RevenueShare"] = (display["RevenueShare"] * 100).map(lambda value: f"{value:.1f}%")
    return display[["Country", "Revenue", "Orders", "Customers", "Items", "RevenueShare"]]


def render_dashboard(df: pd.DataFrame) -> None:
    """Render the main dashboard page."""

    if df.empty:
        st.warning("No data is available for the current filters.")
        return

    summary = utils.dataset_summary(df)
    kpis = utils.get_kpis(df)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Shopper Spectrum v2.0</div>
            <h1>Retail performance dashboard</h1>
            <p>
                Explore revenue, customer behavior, country mix, and monthly sales
                through a polished Streamlit experience built for final-year project
                presentation and portfolio use.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_metric_grid(summary, kpis)

    st.markdown(
        """
        <div class="toolbar-card">
            <div>
                <div class="toolbar-title">Analysis controls</div>
                <div class="toolbar-subtitle">Tune the ranking windows below to explore the retail mix.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_left, control_right, control_third = st.columns([1.1, 1.1, 0.8])
    with control_left:
        product_metric = st.selectbox(
            "Top products by",
            options=["Revenue", "Quantity"],
            index=0,
            help="Switch between sales value and units sold.",
        )
    with control_right:
        customer_metric = st.selectbox(
            "Top customers by",
            options=["Revenue", "Orders"],
            index=0,
            help="Rank loyal customers by spend or order count.",
        )
    with control_third:
        top_n = st.slider("Top N", min_value=5, max_value=15, value=10, step=1)

    monthly = utils.get_monthly_sales(df)
    st.plotly_chart(_build_monthly_chart(monthly), use_container_width=True)

    products_col, customers_col = st.columns(2)
    with products_col:
        top_products, top_products_display = _prepare_top_products(df, product_metric, top_n)
        st.markdown(
            "<div class='panel-heading'><h3>Top products</h3><p>Ranked by the selected metric.</p></div>",
            unsafe_allow_html=True,
        )
        if top_products.empty:
            st.info("No products are available for the selected filters.")
        else:
            metric_column = product_metric
            top_products_chart = top_products.copy()
            top_products_chart["Description"] = top_products_chart["Description"].map(
                utils.display_text
            )
            chart = _build_bar_chart(
                top_products_chart.rename(columns={metric_column: "Value"}),
                x="Value",
                y="Description",
                title=f"Top products by {product_metric.lower()}",
                color=PRIMARY_COLOR,
                value_format="%{x:,.2f}",
                currency_symbol="₹" if metric_column == "Revenue" else None,
            )
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(top_products_display, use_container_width=True, hide_index=True)

    with customers_col:
        top_customers, top_customers_display = _prepare_top_customers(df, customer_metric, top_n)
        st.markdown(
            "<div class='panel-heading'><h3>Top customers</h3><p>Spending leaders and repeat buyers.</p></div>",
            unsafe_allow_html=True,
        )
        if top_customers.empty:
            st.info("No customers are available for the selected filters.")
        else:
            metric_column = customer_metric
            customer_chart_frame = top_customers.copy()
            customer_chart_frame["CustomerLabel"] = customer_chart_frame["CustomerID"].map(
                utils.format_customer_id
            )
            chart = _build_bar_chart(
                customer_chart_frame.rename(columns={metric_column: "Value"}),
                x="Value",
                y="CustomerLabel",
                title=f"Top customers by {customer_metric.lower()}",
                color=SECONDARY_COLOR,
                value_format="%{x:,.2f}",
                currency_symbol="₹" if metric_column == "Revenue" else None,
            )
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(top_customers_display, use_container_width=True, hide_index=True)

    country_summary = utils.get_country_summary(df, top_n=min(15, summary.countries))
    country_col, table_col = st.columns([1.1, 0.9])
    with country_col:
        st.markdown(
            "<div class='panel-heading'><h3>Country analysis</h3><p>Geographic mix of revenue and customer demand.</p></div>",
            unsafe_allow_html=True,
        )
        if country_summary.empty:
            st.info("Country analysis is unavailable for the current filters.")
        else:
            st.plotly_chart(_build_country_choropleth(country_summary), use_container_width=True)

    with table_col:
        st.markdown(
            "<div class='panel-heading'><h3>Country leaderboard</h3><p>Top markets by revenue share.</p></div>",
            unsafe_allow_html=True,
        )
        if country_summary.empty:
            st.info("No country breakdown available.")
        else:
            st.dataframe(
                _format_country_table(country_summary),
                use_container_width=True,
                hide_index=True,
            )
