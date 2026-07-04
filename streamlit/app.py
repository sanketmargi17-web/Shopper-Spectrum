from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dashboard  # noqa: E402
import recommendation  # noqa: E402
import segmentation  # noqa: E402
import utils  # noqa: E402


PAGE_OPTIONS = [
    "Dashboard",
    "Customer Segmentation",
    "Product Recommendation",
]


def _render_footer() -> None:
    st.markdown(
        """
        <div style="text-align:center; margin-top:2.5rem; padding:1.15rem 0 0.25rem; color:#64748b; line-height:1.7; font-size:0.92rem;">
            <div><strong>Developed by</strong></div>
            <div><strong>Sanket Subhash Margi</strong></div>
            <div>Shopper Spectrum v2.0</div>
            <div>AI Powered Customer Segmentation &amp; Product Recommendation System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(summary: utils.DatasetSummary, df) -> tuple[str, tuple, list[str]]:
    with st.sidebar:
        logo_path = utils.BASE_DIR / "assets" / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path))

        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand__eyebrow">Shopper Spectrum</div>
                <div class="sidebar-brand__title">v2.0</div>
                <div class="sidebar-brand__subtitle">Retail intelligence dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.radio(
            "Navigation",
            options=PAGE_OPTIONS,
            index=0,
            key="page_selection",
        )

        st.markdown(
            f"""
            <div class="sidebar-summary">
                <div><span>Rows</span><strong>{summary.rows:,}</strong></div>
                <div><span>Customers</span><strong>{summary.customers:,}</strong></div>
                <div><span>Products</span><strong>{summary.products:,}</strong></div>
                <div><span>Countries</span><strong>{summary.countries:,}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        date_start = summary.min_date.date() if summary.min_date is not None else None
        date_end = summary.max_date.date() if summary.max_date is not None else None

        if date_start is not None and date_end is not None:
            selected_dates = st.date_input(
                "Invoice date range",
                value=(date_start, date_end),
                min_value=date_start,
                max_value=date_end,
            )
        else:
            selected_dates = None

        country_options = sorted(df["Country"].dropna().astype(str).unique().tolist())
        selected_countries = st.multiselect(
            "Countries",
            options=country_options,
            default=country_options,
            help="Use the sidebar controls to narrow the analysis cohort.",
        )

        st.caption(
            "The recommendation page always uses the full product catalog, while the dashboard and RFM views respect the selected filters."
        )

    page = st.session_state.get("page_selection", PAGE_OPTIONS[0])
    return page, selected_dates, selected_countries


def main() -> None:
    st.set_page_config(
        page_title="Shopper Spectrum v2.0",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    utils.inject_styles()

    try:
        full_df = utils.load_clean_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(f"Dataset preparation failed: {exc}")
        st.stop()
    except Exception as exc:  # pragma: no cover - startup guard
        st.error(f"Unable to load the retail dataset: {exc}")
        st.stop()

    summary = utils.dataset_summary(full_df)
    page, selected_dates, selected_countries = _render_sidebar(summary, full_df)

    filtered_df = utils.apply_filters(
        full_df,
        date_range=selected_dates,
        countries=selected_countries if selected_countries else None,
    )

    if page == "Dashboard":
        dashboard.render_dashboard(filtered_df)
    elif page == "Customer Segmentation":
        segmentation.render_segmentation(filtered_df, full_df)
    elif page == "Product Recommendation":
        recommendation.render_recommendation(full_df)
    else:
        st.error("Unknown page selection.")

    _render_footer()


if __name__ == "__main__":
    main()
