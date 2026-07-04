from __future__ import annotations

import html
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils


SEGMENT_COLORS = {
    "High Value": "#0f766e",
    "Regular": "#2563eb",
    "Occasional": "#f59e0b",
    "At-Risk": "#dc2626",
}

SEGMENT_BADGES = {
    "High Value": "🟢",
    "Regular": "🔵",
    "Occasional": "🟠",
    "At-Risk": "🔴",
}


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-kicker">RFM Analysis</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(title: str, value: str, caption: str, tone: str = "primary") -> str:
    return f"""
    <div class="metric-card metric-card--{tone}">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


def _render_metrics(segmented: pd.DataFrame, summary: pd.DataFrame) -> None:
    total_customers = len(segmented)
    total_segments = int(summary["Segment"].nunique()) if not summary.empty else 0
    high_value_count = int((segmented["Segment"] == "High Value").sum()) if "Segment" in segmented.columns else 0
    at_risk_count = int((segmented["Segment"] == "At-Risk").sum()) if "Segment" in segmented.columns else 0
    average_monetary = float(segmented["Monetary"].mean()) if not segmented.empty else 0.0

    cards = [
        _metric_card("Customers analysed", utils.format_number(total_customers), "Customers in the current cohort", "teal"),
        _metric_card("Segments present", utils.format_number(total_segments), "Clusters detected in the view", "blue"),
        _metric_card("High value", utils.format_number(high_value_count), "Recent, frequent, and high spending", "amber"),
        _metric_card("At risk", utils.format_number(at_risk_count), "Low activity customers", "indigo"),
        _metric_card("Avg. monetary", utils.format_currency(average_monetary), "Spend per customer", "slate"),
    ]

    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)


def _build_scatter(segmented: pd.DataFrame) -> go.Figure:
    plot_frame = segmented.copy()
    plot_frame["CustomerIDLabel"] = plot_frame.index.map(utils.format_customer_id)
    figure = px.scatter(
        plot_frame,
        x="Recency",
        y="Monetary",
        size="Frequency",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="CustomerIDLabel",
        custom_data=["Cluster", "Recency", "Frequency", "Monetary"],
        log_y=True,
        size_max=18,
        opacity=0.88,
    )
    figure.update_traces(
        hovertemplate=(
            "Customer: %{hovertext}<br>"
            "Cluster: %{customdata[0]}<br>"
            "Recency: %{customdata[1]} days<br>"
            "Frequency: %{customdata[2]} orders<br>"
            "Monetary: ₹ %{customdata[3]:,.2f}"
            "<extra></extra>"
        )
    )
    figure.update_layout(
        title="Customer distribution by Recency and Monetary value",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=470,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#0f172a"),
    )
    figure.update_xaxes(title_text="Recency (days)", gridcolor="#e2e8f0")
    figure.update_yaxes(title_text="Monetary (₹, log scale)", gridcolor="#e2e8f0")
    return figure


def _build_segment_bar(summary: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        summary,
        x="Segment",
        y="Customers",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        text="Customers",
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_layout(
        title="Customer count by segment",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=470,
        showlegend=False,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#0f172a"),
    )
    figure.update_yaxes(title_text="Customers", gridcolor="#e2e8f0")
    figure.update_xaxes(title_text="")
    return figure


def _format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    display["Customers"] = display["Customers"].map(utils.format_number)
    display["AvgRecency"] = display["AvgRecency"].round(1)
    display["AvgFrequency"] = display["AvgFrequency"].round(1)
    display["AvgMonetary"] = display["AvgMonetary"].map(utils.format_currency)
    display["Share"] = (display["Share"] * 100).map(lambda value: f"{value:.1f}%")
    display["RevenueShare"] = (display["RevenueShare"] * 100).map(lambda value: f"{value:.1f}%")
    return display[
        [
            "Cluster",
            "Segment",
            "Customers",
            "AvgRecency",
            "AvgFrequency",
            "AvgMonetary",
            "Share",
            "RevenueShare",
        ]
    ]


def _format_cluster_centers(bundle: utils.SegmentModelBundle) -> pd.DataFrame:
    display = bundle.cluster_centers.copy()
    display["Recency"] = display["Recency"].round(1)
    display["Frequency"] = display["Frequency"].round(1)
    display["Monetary"] = display["Monetary"].map(utils.format_currency)
    return display[["Cluster", "Label", "Recency", "Frequency", "Monetary"]]


def _customer_detail_card(row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="detail-card">
            <div class="detail-title">Customer {utils.format_customer_id(row.name)}</div>
            <div class="detail-grid">
                <div><span>Segment</span><strong>{row["Segment"]}</strong></div>
                <div><span>Cluster</span><strong>{int(row["Cluster"])}</strong></div>
                <div><span>Recency</span><strong>{int(row["Recency"])} days</strong></div>
                <div><span>Frequency</span><strong>{int(row["Frequency"])} orders</strong></div>
                <div><span>Monetary</span><strong>{utils.format_currency(row["Monetary"])}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prediction_success_card(row: pd.Series) -> None:
    segment = utils.clean_text(row["Segment"])
    badge = SEGMENT_BADGES.get(segment, "🟦")

    st.markdown(
        dedent(
            f"""
            <div style="margin:0.9rem 0 1.1rem; padding:1.1rem 1.2rem; border-radius:20px; border:1px solid rgba(34,197,94,0.22); background: linear-gradient(180deg, rgba(240,253,244,0.98), rgba(236,253,245,0.92)); box-shadow: 0 16px 34px rgba(15,23,42,0.08);">
                <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:0.72rem; font-weight:800; color:#166534;">🎯 Predicted Customer Segment</div>
                <div style="margin-top:0.5rem; font-size:1.32rem; font-weight:800; color:#14532d;">{badge} {html.escape(segment)} Customer</div>
                <div style="margin-top:1rem; display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:0.8rem;">
                    <div style="border-radius:14px; background:rgba(255,255,255,0.76); border:1px solid rgba(34,197,94,0.14); padding:0.82rem 0.9rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b; font-weight:700;">Cluster Number</div>
                        <div style="margin-top:0.35rem; font-size:1.08rem; font-weight:800; color:#0f172a;">{int(row["Cluster"])}</div>
                    </div>
                    <div style="border-radius:14px; background:rgba(255,255,255,0.76); border:1px solid rgba(34,197,94,0.14); padding:0.82rem 0.9rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b; font-weight:700;">Recency</div>
                        <div style="margin-top:0.35rem; font-size:1.08rem; font-weight:800; color:#0f172a;">{int(row["Recency"])} days</div>
                    </div>
                    <div style="border-radius:14px; background:rgba(255,255,255,0.76); border:1px solid rgba(34,197,94,0.14); padding:0.82rem 0.9rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b; font-weight:700;">Frequency</div>
                        <div style="margin-top:0.35rem; font-size:1.08rem; font-weight:800; color:#0f172a;">{int(row["Frequency"])} orders</div>
                    </div>
                    <div style="border-radius:14px; background:rgba(255,255,255,0.76); border:1px solid rgba(34,197,94,0.14); padding:0.82rem 0.9rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b; font-weight:700;">Monetary</div>
                        <div style="margin-top:0.35rem; font-size:1.08rem; font-weight:800; color:#0f172a;">{html.escape(utils.format_currency(row["Monetary"]))}</div>
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def _prediction_result_card(row: pd.Series) -> None:
    segment = utils.clean_text(row["Segment"])
    normalized_segment = segment.lower().replace("-", " ").replace("  ", " ").strip()
    display_segment = segment.replace("-", " ")

    card_configs = {
        "high value": {
            "title": "🟢 High Value Customer",
            "accent": "#0f766e",
            "border": "#0f766e",
            "tint": "rgba(15, 118, 110, 0.08)",
            "surface": "rgba(15, 118, 110, 0.14)",
            "border_soft": "rgba(15, 118, 110, 0.18)",
            "highlights": ["✅ Loyal Customer", "💰 High Spending", "🛒 Frequent Purchaser"],
            "actions": ["Premium Offers", "VIP Membership", "Early Product Launch"],
        },
        "regular": {
            "title": "🟡 Regular Customer",
            "accent": "#ca8a04",
            "border": "#ca8a04",
            "tint": "rgba(202, 138, 4, 0.08)",
            "surface": "rgba(202, 138, 4, 0.14)",
            "border_soft": "rgba(202, 138, 4, 0.18)",
            "highlights": ["✅ Moderate Spending", "📦 Average Purchase Frequency"],
            "actions": ["Bundle Offers", "Loyalty Rewards", "Upselling Campaign"],
        },
        "at risk": {
            "title": "🔴 At Risk Customer",
            "accent": "#dc2626",
            "border": "#dc2626",
            "tint": "rgba(220, 38, 38, 0.08)",
            "surface": "rgba(220, 38, 38, 0.14)",
            "border_soft": "rgba(220, 38, 38, 0.18)",
            "highlights": ["⚠ Low Recent Activity", "📉 Reduced Purchase Frequency"],
            "actions": ["Discount Coupons", "Win-back Email", "Retention Campaign"],
        },
        "occasional": {
            "title": "🔵 Occasional Customer",
            "accent": "#2563eb",
            "border": "#2563eb",
            "tint": "rgba(37, 99, 235, 0.08)",
            "surface": "rgba(37, 99, 235, 0.14)",
            "border_soft": "rgba(37, 99, 235, 0.18)",
            "highlights": ["🛍 Purchases Occasionally", "📈 Growth Opportunity"],
            "actions": ["Product Recommendations", "Seasonal Offers", "Loyalty Program"],
        },
    }

    config = card_configs.get(normalized_segment)
    if config is None:
        config = {
            "title": f"📌 {display_segment} Customer",
            "accent": "#475569",
            "border": "#64748b",
            "tint": "rgba(100, 116, 139, 0.08)",
            "surface": "rgba(100, 116, 139, 0.14)",
            "border_soft": "rgba(100, 116, 139, 0.18)",
            "highlights": [f"Segment: {display_segment}"],
            "actions": ["Review the customer profile", "Tailor engagement", "Monitor activity"],
        }

    st.markdown(
        dedent(
            f"""
            <div style="margin:0.9rem 0 1.1rem; padding:1.15rem 1.25rem; border-radius:22px; border:1px solid {config["border_soft"]}; border-left:8px solid {config["border"]}; background:linear-gradient(180deg, {config["tint"]}, rgba(255,255,255,0.94)); box-shadow:0 16px 34px rgba(15,23,42,0.08); font-family:'Trebuchet MS', 'Segoe UI', sans-serif;">
                <div style="display:flex; flex-wrap:wrap; align-items:flex-start; justify-content:space-between; gap:0.75rem;">
                    <div style="min-width:0;">
                        <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:0.72rem; font-weight:800; color:{config["accent"]};">Predicted Customer Segment</div>
                        <div style="margin-top:0.45rem; font-size:1.36rem; font-weight:800; line-height:1.25; color:#0f172a;">{html.escape(config["title"])}</div>
                    </div>
                    <div style="padding:0.48rem 0.85rem; border-radius:999px; border:1px solid {config["border_soft"]}; background:{config["surface"]}; color:{config["accent"]}; font-size:0.82rem; font-weight:800; letter-spacing:0.02em; white-space:nowrap;">Status: {html.escape(display_segment)}</div>
                </div>
                <div style="margin-top:1rem; display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:0.85rem;">
                    <div style="border-radius:16px; background:rgba(255,255,255,0.82); border:1px solid {config["border_soft"]}; padding:0.95rem 1rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b; font-weight:800;">Customer Signals</div>
                        <ul style="margin:0.7rem 0 0; padding-left:1.15rem; color:#334155; font-size:0.98rem; line-height:1.75;">
                            {''.join(f'<li>{html.escape(item)}</li>' for item in config["highlights"])}
                        </ul>
                    </div>
                    <div style="border-radius:16px; background:rgba(255,255,255,0.82); border:1px solid {config["border_soft"]}; padding:0.95rem 1rem;">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b; font-weight:800;">Recommended Action</div>
                        <ul style="margin:0.7rem 0 0; padding-left:1.15rem; color:#334155; font-size:0.98rem; line-height:1.75;">
                            {''.join(f'<li>{html.escape(item)}</li>' for item in config["actions"])}
                        </ul>
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_segmentation(
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> None:
    """Render the RFM analysis and customer segmentation page."""

    if filtered_df.empty:
        st.warning("No transactions are available for the current filters.")
        return

    try:
        bundle = utils.load_segmentation_bundle(full_df)
        rfm = utils.build_rfm_table(filtered_df)
        segmented, summary = utils.assign_customer_segments(rfm, bundle)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unable to complete segmentation: {exc}")
        return

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Customer Segmentation</div>
            <h1>RFM-based customer intelligence</h1>
            <p>
                The app scores customers by recency, frequency, and monetary value,
                then applies the saved scaler and KMeans model to surface actionable
                customer groups.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_metrics(segmented, summary)

    controls_left, controls_right = st.columns([1.1, 1.2])
    with controls_left:
        segment_options = ["All segments"] + [bundle.label_map[cluster] for cluster in bundle.ordered_clusters]
        segment_focus = st.selectbox(
            "Focus segment",
            options=segment_options,
            help="Filter the tables to a single customer segment.",
        )
    with controls_right:
        max_rows = max(1, min(50, len(segmented)))
        row_limit = st.slider(
            "Table rows",
            min_value=1,
            max_value=max_rows,
            value=min(15, max_rows),
            step=1,
            help="Limit the number of customers shown in the preview table.",
        )

    if segment_focus != "All segments":
        working = segmented.loc[segmented["Segment"] == segment_focus].copy()
    else:
        working = segmented.copy()

    if working.empty:
        st.warning("No customers matched the selected segment filter.")
        return

    chart_col, bar_col = st.columns(2)
    with chart_col:
        st.plotly_chart(_build_scatter(working), use_container_width=True)
    with bar_col:
        st.plotly_chart(_build_segment_bar(summary), use_container_width=True)

    center_col, summary_col = st.columns([1.05, 1.15])
    with center_col:
        st.markdown(
            "<div class='panel-heading'><h3>Model cluster centers</h3><p>Inverse-transformed centroids from the saved scaler and KMeans model.</p></div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            _format_cluster_centers(bundle),
            use_container_width=True,
            hide_index=True,
        )
    with summary_col:
        st.markdown(
            "<div class='panel-heading'><h3>Segment summary</h3><p>Aggregated from the current filtered cohort.</p></div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            _format_summary_table(summary),
            use_container_width=True,
            hide_index=True,
        )

    customer_ids = working.index.tolist()
    selected_customer = st.selectbox(
        "Inspect customer",
        options=customer_ids,
        format_func=utils.format_customer_id,
    )
    predict_clicked = st.button("Predict", type="primary")
    if predict_clicked and selected_customer in working.index:
        _prediction_result_card(working.loc[selected_customer])

    if selected_customer in working.index:
        _customer_detail_card(working.loc[selected_customer])

    st.markdown(
        "<div class='panel-heading'><h3>Customer-level RFM table</h3><p>Preview of the segmented cohort sorted by monetary value.</p></div>",
        unsafe_allow_html=True,
    )
    preview = working.reset_index().rename(columns={"index": "CustomerID"}).head(row_limit)
    preview["CustomerID"] = preview["CustomerID"].map(utils.format_customer_id)
    preview["Recency"] = preview["Recency"].map(utils.format_number)
    preview["Frequency"] = preview["Frequency"].map(utils.format_number)
    preview["Monetary"] = preview["Monetary"].map(utils.format_currency)
    st.dataframe(
        preview[["CustomerID", "Segment", "Cluster", "Recency", "Frequency", "Monetary"]],
        use_container_width=True,
        hide_index=True,
    )
