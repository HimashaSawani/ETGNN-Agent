import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ETGNN Research Demo", layout="wide")
st.title("Elliptic Transaction Research Demo")
st.caption(
    "Predictions for transactions already present in the Elliptic dataset. "
    "Uses completed time-step graph information; this is not live detection."
)

api_url = st.sidebar.text_input(
    "API URL", value="http://127.0.0.1:8000"
).rstrip("/")

try:
    response = requests.get(f"{api_url}/examples", timeout=180)
    response.raise_for_status()
    examples = response.json()
except requests.RequestException as error:
    st.error(f"Cannot connect to the FastAPI backend: {error}")
    st.stop()

if examples:
    choices = {
        f"{item['case']} - transaction {item['transaction_id']} "
        f"(step {item['time_step']})": item["transaction_id"]
        for item in examples
    }
    mode = st.radio(
        "Input source",
        ["Saved explanation case", "Transaction ID"],
        horizontal=True,
    )
else:
    choices = {}
    mode = "Transaction ID"
    st.info("No saved explanation examples were found.")

if mode == "Saved explanation case":
    selected = st.selectbox("Choose a case", options=list(choices))
    transaction_id = choices[selected]
else:
    transaction_id = st.number_input(
        "Elliptic transaction ID",
        min_value=1,
        value=230425980,
        step=1,
        format="%d",
    )

if st.button("Analyze transaction", type="primary"):
    try:
        with st.spinner("Loading prediction..."):
            response = requests.get(
                f"{api_url}/transactions/{int(transaction_id)}",
                timeout=180,
            )
        if response.status_code == 404:
            st.warning("This transaction ID is not in the Elliptic dataset.")
            st.stop()
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")
        st.stop()

    st.subheader(f"Transaction {data['transaction_id']}")
    st.write(
        f"Time step: **{data['time_step']}** | "
        f"Dataset split: **{data['split']}**"
    )
    if data["actual_class"] is not None:
        actual = "Illicit" if data["actual_class"] == 1 else "Licit"
        st.write(f"Dataset label: **{actual}**")
    else:
        st.write("Dataset label: **Unknown**")

    left, right = st.columns(2)
    for column, title, key in [
        (left, "Random Forest", "random_forest"),
        (right, "ETGNN", "etgnn"),
    ]:
        result = data[key]
        with column:
            st.subheader(title)
            st.metric(
                "Illicit probability",
                f"{result['illicit_probability']:.1%}",
            )
            predicted = (
                "Illicit" if result["predicted_class"] == 1
                else "Licit"
            )
            st.write(f"Predicted class: **{predicted}**")

    explanation = data["explanation"]
    st.divider()
    st.subheader("ETGNN explanation")
    if explanation is None:
        st.info(
            "No saved GNNExplainer result for this transaction. "
            "Choose one of the example transactions above."
        )
    else:
        st.write(f"Selected validation case: **{explanation['case']}**")
        st.write(
            f"Two-hop subgraph: {explanation['subgraph_nodes']} nodes, "
            f"{explanation['subgraph_edges']} edges"
        )
        st.write("Top anonymous feature column indices:")
        feature_rows = [
            {"column_index": index, "mask_value": value}
            for index, value in zip(
                explanation["top_feature_columns_zero_based"],
                explanation["top_feature_mask_values"],
            )
        ]
        st.dataframe(pd.DataFrame(feature_rows), hide_index=True,
                     use_container_width=True)
        st.write("Top ranked graph edges:")
        st.dataframe(pd.DataFrame(explanation["top_edges"]), hide_index=True, use_container_width=True)
        st.caption(
            "Mask values describe this model's behavior. They are not "
            "verified causal explanations; feature names are anonymous."
        )

