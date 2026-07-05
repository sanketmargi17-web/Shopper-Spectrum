
from __future__ import annotations
import streamlit as st


import html
from textwrap import dedent

import pandas as pd
import plotly.express as px
import streamlit as st

import utils


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-kicker">Recommendation Engine</div>
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


def _format_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if display.empty:
        return display
    display["Product"] = display["Product"].map(utils.display_text)
    display["Similarity"] = display["Similarity"].map(lambda value: f"{float(value):.4f}")
    return display[["Rank", "Product", "Similarity"]]


def _render_recommendation_cards(frame: pd.DataFrame, limit: int = 5) -> None:
    display = _format_recommendations(frame).head(limit)
    if display.empty:
        return

    rank_badges = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    card_html: list[str] = []

    for _, row in display.iterrows():
        rank = int(row["Rank"]) if pd.notna(row["Rank"]) else len(card_html) + 1
        badge = rank_badges[rank - 1] if 1 <= rank <= len(rank_badges) else f"{rank}."
        card_html.append(
            dedent(
                f"""
                <div style="border:1px solid rgba(15,118,110,0.14); border-radius:18px; padding:0.95rem 1rem; background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(240,253,250,0.96)); box-shadow: 0 16px 34px rgba(15,23,42,0.08);">
                    <div style="display:flex; align-items:flex-start; gap:0.85rem;">
                        <div style="font-size:1.25rem; line-height:1; flex:0 0 auto;">{badge}</div>
                        <div style="flex:1; min-width:0;">
                            <div style="font-weight:800; color:#0f172a; line-height:1.4;">{html.escape(utils.product_label(row["Product"]))}</div>
                            <div style="margin-top:0.25rem; font-size:0.86rem; color:#64748b;">Similarity score: {html.escape(str(row["Similarity"]))}</div>
                        </div>
                    </div>
                </div>
                """
            ).strip()
        )

    st.markdown(
        dedent(
            f"""
            <div style="display:grid; gap:0.8rem; margin:0.25rem 0 1rem;">
                {''.join(card_html)}
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def _build_similarity_chart(frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    chart_frame = frame.copy()
    chart_frame["Product"] = chart_frame["Product"].map(utils.display_text)
    chart = px.bar(
        chart_frame.sort_values("Similarity", ascending=True),
        x="Similarity",
        y="Product",
        orientation="h",
        color="Similarity",
        color_continuous_scale=["#cffafe", "#14b8a6", "#0f766e"],
        text="Similarity",
    )
    chart.update_traces(
        texttemplate="%{x:.3f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="Product: %{y}<br>Similarity: %{x:.4f}<extra></extra>",
    )
    chart.update_layout(
        title="Similarity ranking",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
        coloraxis_showscale=False,
        showlegend=False,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#0f172a"),
    )
    chart.update_xaxes(title_text="Cosine similarity", gridcolor="#e2e8f0")
    chart.update_yaxes(title_text="")
    st.plotly_chart(chart, use_container_width=True)


def render_recommendation(full_df: pd.DataFrame) -> None:
    """Render the product recommendation page."""

    if full_df.empty:
        st.warning("No product catalog is available for recommendations.")
        return

    try:
        store = utils.load_recommendation_store()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unable to load recommendation artifacts: {exc}")
        return

    summary = utils.dataset_summary(full_df)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Product Recommendation</div>
            <h1>Similar products powered by cosine similarity</h1>
            <p>
                Search the catalog, choose a product, and the model will surface the
                closest matches from the saved similarity matrix.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        _metric_card("Catalog products", utils.format_number(summary.products), "Unique descriptions in the dataset", "teal"),
        _metric_card("Similarity type", store.source_type, "Loaded from the saved artifact", "blue"),
        _metric_card(
            "Matrix size",
            f"{len(store.product_names):,} x {len(store.product_names):,}",
            "Similarity dimensions",
            "amber",
        ),
    ]
    metric_columns = st.columns(len(cards))
    for column, card in zip(metric_columns, cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)

    query_col, product_col, topn_col = st.columns([1.2, 1.8, 0.8])
    with query_col:
        query = st.text_input(
            "Search products",
            value="",
            placeholder="Type part of a product name",
        )

    matches = utils.search_products(store.product_names, query, limit=250)
    if not matches:
        st.warning("No products matched the current search term.")
        return

    with product_col:
        selected_product = st.selectbox(
            "Choose a product",
            options=matches,
            format_func=utils.product_label,
        )
    with topn_col:
        top_n = st.slider(
            "Recommendations",
            min_value=3,
            max_value=12,
            value=5,
            step=1,
            help="Choose how many similar products to display.",
        )

    recommendations = utils.recommend_products(selected_product, store, top_n=top_n)
    if recommendations.empty:
        st.info("No recommendations were found for the selected product.")
        return

    selected_row = full_df.loc[
        full_df["Description"].astype(str).str.strip() == utils.product_key(selected_product)
    ]
    selected_revenue = float(selected_row["TotalAmount"].sum()) if not selected_row.empty else 0.0
    selected_orders = int(selected_row["InvoiceNo"].nunique()) if not selected_row.empty else 0

    detail_left, detail_right = st.columns([1.1, 0.9])
    with detail_left:
        st.markdown(
            f"""
            <div class="detail-card">
                <div class="detail-title">{utils.product_label(selected_product)}</div>
                <div class="detail-grid">
                    <div><span>Catalog appearances</span><strong>{utils.format_number(len(selected_row))}</strong></div>
                    <div><span>Orders containing it</span><strong>{utils.format_number(selected_orders)}</strong></div>
                    <div><span>Revenue generated</span><strong>{utils.format_currency(selected_revenue)}</strong></div>
                    <div><span>Similarity source</span><strong>{store.source_type}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with detail_right:
        st.markdown(
            """
            <div class="panel-heading">
                <h3>How the recommender works</h3>
                <p>
                    A cosine similarity matrix captures co-purchase patterns across the
                    product catalog. The app handles both DataFrame and NumPy matrix
                    artifacts automatically.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='panel-heading'><h3>✅ Top 5 Recommended Products</h3><p>Closest matching items from the similarity model.</p></div>",
        unsafe_allow_html=True,
    )
    _render_recommendation_cards(recommendations, limit=5)
    st.dataframe(
        _format_recommendations(recommendations),
        use_container_width=True,
        hide_index=True,
    )
    _build_similarity_chart(recommendations)
