import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.predict import predict_price


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diamond Price Estimator",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hero section */
    .hero {
        padding: 2rem 2rem 2.2rem 2rem;
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 18px;
        margin-bottom: 2rem;
        background: linear-gradient(
            135deg,
            rgba(100, 100, 100, 0.08),
            rgba(255, 255, 255, 0.02)
        );
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 0.7rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        line-height: 1.6;
        opacity: 0.78;
        max-width: 850px;
    }

    /* Section headings */
    .section-title {
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .section-description {
        opacity: 0.72;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    .metric-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 14px;
        padding: 1.2rem;
        min-height: 125px;
    }

    .metric-label {
        font-size: 0.85rem;
        opacity: 0.65;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-caption {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-top: 0.25rem;
    }

    /* Prediction card */
    .prediction-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .prediction-label {
        font-size: 0.9rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .prediction-value {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.35rem 0;
    }

    .prediction-note {
        opacity: 0.65;
        font-size: 0.85rem;
    }

    /* Info box */
    .info-box {
        border-left: 4px solid currentColor;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-radius: 4px;
        background: rgba(128, 128, 128, 0.07);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 650;
        min-height: 3rem;
    }

    /* Hide default Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

TEST_R2 = 0.9794
TEST_MAE = 449.92
TRAIN_OBSERVATIONS = 43133
TEST_OBSERVATIONS = 10784
SMEARING_CORRECTION = 1.0107

CUT_OPTIONS = [
    "Fair",
    "Good",
    "Very Good",
    "Premium",
    "Ideal",
]

COLOR_OPTIONS = [
    "J",
    "I",
    "H",
    "G",
    "F",
    "E",
    "D",
]

CLARITY_OPTIONS = [
    "I1",
    "SI2",
    "SI1",
    "VS2",
    "VS1",
    "VVS2",
    "VVS1",
    "IF",
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def money(value):
    return f"${value:,.0f}"


def render_metric(label, value, caption):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def price_sensitivity_chart(
    base_carat,
    cut,
    color,
    clarity,
    depth,
    table,
):
    """
    Generate predictions across a range of carat values while
    keeping the remaining inputs fixed.
    """

    lower = max(0.2, base_carat * 0.4)
    upper = min(5.0, max(base_carat * 1.6, base_carat + 0.5))

    carat_values = np.linspace(lower, upper, 40)

    predictions = []

    for carat in carat_values:
        price = predict_price(
            carat=float(carat),
            cut=cut,
            color=color,
            clarity=clarity,
            depth=depth,
            table=table,
        )
        predictions.append(price)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(carat_values, predictions, linewidth=2.5)

    ax.scatter(
        [base_carat],
        [
            predict_price(
                carat=base_carat,
                cut=cut,
                color=color,
                clarity=clarity,
                depth=depth,
                table=table,
            )
        ],
        s=70,
        zorder=5,
    )

    ax.set_title("Estimated Price vs. Carat")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Estimated price ($)")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig

def show_model_plot(filename, caption):
    plot_path = ROOT_DIR / "outputs" / filename

    if plot_path.exists():
        st.image(
            str(plot_path),
            use_container_width=True,
        )
        st.caption(caption)
    else:
        st.warning(
            f"Visualization not found: {filename}. "
            "Run the training script again to generate it."
        )

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 💎 Diamond Price Model")

    st.markdown(
        """
        Estimate a diamond's price using a statistical regression
        model trained on the Seaborn Diamonds dataset.
        """
    )

    st.divider()

    st.markdown("### Model")

    st.markdown(
        """
        **Log-Log OLS Regression**

        Final predictors:

        - log(carat)
        - cut
        - color
        - clarity
        - depth
        - table
        """
    )

    st.divider()

    st.caption(
        "Educational portfolio project · "
        "Predictions are estimates, not professional appraisals."
    )


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💎 Diamond Price Estimator</div>
        <div class="hero-subtitle">
            An interpretable statistical model for estimating diamond prices
            from physical and quality characteristics. The model uses a
            log-log regression approach to capture the non-linear relationship
            between carat weight and price.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------

tab_estimator, tab_insights, tab_methodology = st.tabs(
    [
        "💎 Price Estimator",
        "📊 Model Insights",
        "🔬 Methodology",
    ]
)


# =========================================================
# TAB 1 — PRICE ESTIMATOR
# =========================================================

with tab_estimator:

    st.markdown(
        '<div class="section-title">Estimate a diamond\'s market price</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Enter the diamond's characteristics below. The model estimates
            price using the same preprocessing and feature representation
            used during training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:

        st.markdown("### Physical characteristics")

        carat = st.number_input(
            "Carat weight",
            min_value=0.20,
            max_value=5.00,
            value=1.00,
            step=0.01,
            help="Weight of the diamond in carats.",
        )

        depth = st.number_input(
            "Depth (%)",
            min_value=50.0,
            max_value=75.0,
            value=61.5,
            step=0.1,
            help="Diamond depth as a percentage of its average diameter.",
        )

        table = st.number_input(
            "Table (%)",
            min_value=50.0,
            max_value=70.0,
            value=57.0,
            step=0.1,
            help="Width of the diamond's top facet relative to its widest point.",
        )

    with right:

        st.markdown("### Quality characteristics")

        cut = st.selectbox(
            "Cut",
            CUT_OPTIONS,
            index=CUT_OPTIONS.index("Ideal"),
            help="Overall cut quality.",
        )

        color = st.selectbox(
            "Color",
            COLOR_OPTIONS,
            index=COLOR_OPTIONS.index("G"),
            help="Diamond color grade from D (highest) to J.",
        )

        clarity = st.selectbox(
            "Clarity",
            CLARITY_OPTIONS,
            index=CLARITY_OPTIONS.index("VS1"),
            help="Clarity grade from I1 to IF.",
        )

    st.divider()

    estimate = st.button(
        "💎 Estimate Price",
        type="primary",
        use_container_width=True,
    )

    if estimate:

        predicted_price = predict_price(
            carat=carat,
            cut=cut,
            color=color,
            clarity=clarity,
            depth=depth,
            table=table,
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Price</div>
                <div class="prediction-value">
                    {money(predicted_price)}
                </div>
                <div class="prediction-note">
                    Model estimate based on the characteristics provided
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Typical error — explicitly NOT a confidence interval.
        low = max(0, predicted_price - TEST_MAE)
        high = predicted_price + TEST_MAE

        st.markdown(
            f"""
            <div class="info-box">
                <strong>How should you interpret this?</strong><br>
                The model's held-out test MAE is approximately
                <strong>{money(TEST_MAE)}</strong>. As a rough indication of
                typical prediction error, that places this estimate around
                <strong>{money(low)} – {money(high)}</strong>.
                This is <strong>not</strong> a statistical confidence interval
                or appraisal range.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Price sensitivity")

        st.caption(
            "How the model's estimated price changes with carat weight "
            "while keeping the other selected characteristics fixed."
        )

        fig = price_sensitivity_chart(
            base_carat=carat,
            cut=cut,
            color=color,
            clarity=clarity,
            depth=depth,
            table=table,
        )

        st.pyplot(fig, use_container_width=True)

        st.caption(
            "The non-linear curve reflects the log-log relationship used "
            "by the final regression model."
        )

    else:

        st.info(
            "Enter the diamond characteristics and click **Estimate Price** "
            "to generate a prediction."
        )


# =========================================================
# TAB 2 — MODEL INSIGHTS
# =========================================================

with tab_insights:

    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            The final model was evaluated on a held-out test set rather than
            only on the observations used for fitting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric(
            "Test R²",
            "97.94%",
            "Log-price scale",
        )

    with col2:
        render_metric(
            "Test MAE",
            "$449.92",
            "Mean absolute error",
        )

    with col3:
        render_metric(
            "Training data",
            f"{TRAIN_OBSERVATIONS:,}",
            "Observations",
        )

    with col4:
        render_metric(
            "Test data",
            f"{TEST_OBSERVATIONS:,}",
            "Held-out observations",
        )

    st.markdown("### Why the final model?")

    st.markdown(
        """
        The project started with ordinary linear regression, but diagnostic
        analysis revealed several problems:

        **1. Multicollinearity**

        Carat weight was strongly related to the physical dimensions
        `x`, `y`, and `z`. Keeping all of these predictors created
        substantial redundancy in the linear model.

        **2. Non-linearity**

        Diamond price does not increase proportionally with carat weight.
        Larger stones command disproportionately higher prices.

        **3. Heteroscedasticity**

        The variability of price increased substantially for more expensive
        diamonds, making a simple linear price model less appropriate.

        **4. Log transformation**

        The final model predicts `log(price)` from `log(carat)` together
        with the quality characteristics. This produced a much better
        representation of the underlying relationship.

        **5. Smearing correction**

        Predictions are transformed back from log-price to dollar price
        using a smearing correction to reduce retransformation bias.
        """
    )

    st.markdown("### Original model comparison")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Model A",
                "Model B",
                "Model C — Final",
            ],
            "R²": [
                0.9044,
                0.9054,
                0.9792,
            ],
            "MAE": [
                854.01,
                850.92,
                459.02,
            ],
            "Approach": [
                "Linear regression",
                "Alternative predictor combination",
                "Log-log regression",
            ],
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "R²": st.column_config.NumberColumn(
                "R²",
                format="%.4f",
            ),
            "MAE": st.column_config.NumberColumn(
                "MAE ($)",
                format="$%.2f",
            ),
        },
    )

    st.caption(
        "Model A/B/C figures above are the original full-dataset notebook "
        "comparison. They are included to show the model-selection story; "
        "they should not be interpreted as held-out test metrics."
    )

    st.markdown("### Final model validation")

    validation_df = pd.DataFrame(
        {
            "Metric": [
                "Training observations",
                "Test observations",
                "Test R²",
                "Test MAE",
                "Smearing correction",
                "Random seed",
            ],
            "Value": [
                f"{TRAIN_OBSERVATIONS:,}",
                f"{TEST_OBSERVATIONS:,}",
                f"{TEST_R2:.4f}",
                f"${TEST_MAE:,.2f}",
                f"{SMEARING_CORRECTION:.4f}",
                "42",
            ],
        }
    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Model diagnostics")

    st.markdown(
        """
        These visualizations use the held-out test set to show how the model
        behaves beyond a single summary metric.
        """
        )

    plot_col1, plot_col2 = st.columns(2, gap="large")

    with plot_col1:
        st.markdown("#### Actual vs Predicted")

        show_model_plot(
            "actual_vs_predicted.png",
            "Points closer to the diagonal represent more accurate predictions."
            )

    with plot_col2:
        st.markdown("#### Prediction Error Distribution")

        show_model_plot(
            "prediction_errors.png",
            "Errors are calculated as predicted price minus actual price."
            )

    st.markdown("#### Price and Carat Relationship")

    show_model_plot(
        "price_vs_carat.png",
        "The increasing spread and non-linear relationship between carat and "
        "price motivate the log-log modeling approach."
        )

    st.markdown(
        """
        <div class="info-box">
            <strong>Portfolio takeaway:</strong>
            The key strength of this project is not simply the final R².
            The model was selected after diagnosing multicollinearity,
            non-linearity, and heteroscedasticity, then validated against
            previously unseen observations.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TAB 3 — METHODOLOGY
# =========================================================

with tab_methodology:

    st.markdown(
        '<div class="section-title">From raw data to price estimate</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            The project follows a statistical modeling workflow rather than
            treating the problem as a black-box prediction task.
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "01",
            "Data exploration",
            "Explored distributions, relationships, correlations, and "
            "potential data-quality issues across the diamond attributes.",
        ),
        (
            "02",
            "Data cleaning",
            "Removed invalid observations and extreme measurement-error "
            "cases before fitting the statistical models.",
        ),
        (
            "03",
            "Feature encoding",
            "Converted the ordinal quality variables — cut, color, and "
            "clarity — into ordered numerical representations.",
        ),
        (
            "04",
            "Baseline regression",
            "Started with ordinary least squares regression using the "
            "available predictors.",
        ),
        (
            "05",
            "Model diagnostics",
            "Investigated multicollinearity, heteroscedasticity, and "
            "non-linear relationships affecting the baseline model.",
        ),
        (
            "06",
            "Log-log transformation",
            "Applied log transformations to price and carat to better "
            "capture the non-linear scaling of diamond prices.",
        ),
        (
            "07",
            "Held-out validation",
            "Used an 80/20 train-test split with random state 42 to "
            "evaluate performance on unseen observations.",
        ),
        (
            "08",
            "Retransformation",
            "Applied a smearing correction when converting predicted "
            "log-prices back into dollar prices.",
        ),
    ]

    for number, title, description in steps:

        st.html(
            f"""
            <div style="
                display:flex;
                gap:1rem;
                padding:1.1rem 0;
                border-bottom:1px solid rgba(128,128,128,0.15);
            ">
                <div style="
                    font-size:0.9rem;
                    font-weight:700;
                    opacity:0.5;
                    min-width:35px;
                ">
                    {number}
                </div>

                <div>
                    <div style="
                        font-size:1.05rem;
                        font-weight:700;
                        margin-bottom:0.25rem;
                    ">
                        {title}
                    </div>

                    <div style="opacity:0.7; line-height:1.5;">
                        {description}
                    </div>
                </div>
            </div>
            """
        )

    st.markdown("### Final model specification")

    st.code(
        """log(price) ~ log(carat) + cut + color + clarity + depth + table""",
        language="text",
    )

    st.markdown("### Important limitations")

    st.markdown(
        """
        This model is designed as an educational and portfolio demonstration.

        The underlying dataset does not capture every factor that can affect
        real-world diamond pricing. For example, certification, fluorescence,
        retail channel, provenance, and other market-specific information
        are not represented in the final model.

        Therefore, the application should be viewed as a **statistical price
        estimator**, not as a professional diamond appraisal or valuation
        service.
        """
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Diamond Price Modeling · OLS Regression · Model Diagnostics · "
    "Statistical Learning"
)
