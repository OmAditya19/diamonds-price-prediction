import streamlit as st

from src.predict import predict_price


st.set_page_config(
    page_title="Diamond Price Estimator",
    page_icon="💎",
    layout="centered",
)


st.title("💎 Diamond Price Estimator")

st.markdown(
    """
    Estimate the price of a diamond using an
    interpretable statistical model trained on
    thousands of diamond observations.
    """
)


st.divider()


st.subheader("Diamond characteristics")


col1, col2 = st.columns(2)


with col1:

    carat = st.number_input(
        "Carat",
        min_value=0.20,
        max_value=5.01,
        value=1.00,
        step=0.01,
    )

    cut = st.selectbox(
        "Cut",
        [
            "Fair",
            "Good",
            "Very Good",
            "Premium",
            "Ideal",
        ],
        index=4,
    )

    color = st.selectbox(
        "Color",
        [
            "J",
            "I",
            "H",
            "G",
            "F",
            "E",
            "D",
        ],
        index=3,
    )


with col2:

    clarity = st.selectbox(
        "Clarity",
        [
            "I1",
            "SI2",
            "SI1",
            "VS2",
            "VS1",
            "VVS2",
            "VVS1",
            "IF",
        ],
        index=4,
    )

    depth = st.number_input(
        "Depth (%)",
        min_value=43.0,
        max_value=79.0,
        value=61.5,
        step=0.1,
    )

    table = st.number_input(
        "Table (%)",
        min_value=43.0,
        max_value=95.0,
        value=57.0,
        step=0.1,
    )


st.divider()


if st.button(
    "Estimate Price",
    type="primary",
    use_container_width=True,
):

    price = predict_price(
        carat=carat,
        cut=cut,
        color=color,
        clarity=clarity,
        depth=depth,
        table=table,
    )

    st.success("Prediction generated")

    st.metric(
        "Estimated Diamond Price",
        f"${price:,.0f}",
    )


st.divider()


st.caption(
    """
    Model note: This estimator is an educational demonstration
    based on the Seaborn Diamonds dataset. It should not be
    interpreted as a professional diamond appraisal or live
    market valuation.
    """
)
