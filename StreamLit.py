import streamlit as st
import pandas as pd
import plotly.express as px
from functions import group_and_aggregate_data  # Ensure these exist
from PCA import dimensionality_reduction  # Ensure these exist


def display_pca_data(df: pd.DataFrame, num_components: int) -> None:
    """
    Displays PCA-reduced data in either 1D, 2D, or 3D using Plotly.
    If the number of components is outside [1, 3],
    the function displays a warning and shows the raw PCA DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that includes columns named 'PC1', 'PC2', 'PC3', ...
    num_components : int
        The number of principal components to display.
    """
    if num_components == 1:
        fig = px.scatter(
            df,
            x='PC1',
            y=[0] * len(df),
            title="1D PCA Visualization",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif num_components == 2:
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            title="2D PCA Visualization",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif num_components == 3:
        fig = px.scatter_3d(
            df,
            x='PC1',
            y='PC2',
            z='PC3',
            title="3D PCA Visualization",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Number of components is outside the recommended range [1, 3]. "
            "Below is the raw PCA DataFrame instead."
        )

    st.write("### PCA Output DataFrame")
    st.dataframe(df)


def main():
    # Set page config
    st.set_page_config(page_title="Dimensionality Reduction", layout="wide")

    with st.sidebar:
        st.title("Data & Settings")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        df = None  # Initialize df as None

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to load the file. Error: {e}")
                st.stop()

            # Column and aggregation
            if df is not None and not df.empty:
                column_names = list(df.columns)
                selected_column = st.selectbox(
                    "Select a column to group by",
                    options=["(No Selection)"] + column_names
                )

                agg_funcs = ['sum', 'mean', 'median', 'min', 'max', 'count', 'std']
                selected_agg = st.selectbox(
                    "Select an aggregation function",
                    options=["(No Selection)"] + agg_funcs
                )

                num_components_input = st.text_input(
                    "Number of PCA Components (1 to 3 recommended)",
                    value="2"
                )

                selection_mode = st.radio(
                    "Data Display Mode",
                    ['City-wise', 'Party-wise'],
                    index=0
                )

                # Validate number of components
                try:
                    num_components = int(num_components_input)
                    if num_components < 1:
                        st.warning("Number of components must be at least 1. Setting to 1.")
                        num_components = 1
                except ValueError:
                    st.warning("Invalid number. Defaulting to 2D PCA.")
                    num_components = 2

                # Prepare button
                valid_column = (selected_column != "(No Selection)")
                valid_agg = (selected_agg != "(No Selection)")
                run_button = st.button("Calculate & Display", disabled=not (valid_column and valid_agg))

    st.title("Dimensionality Reduction App")

    if df is None or df.empty:
        st.info("Please upload a valid CSV or Excel file in the sidebar to begin.")
        return

    # If data is valid and the user clicked the button
    tab1, tab2 = st.tabs(["Data Preview", "PCA Visualization"])

    with tab1:
        st.subheader("1. Preview Original Data")
        st.dataframe(df)

    with tab2:
        st.subheader("2. PCA Results & Visualization")
        if run_button:
            # Group and aggregate
            try:
                aggregated_df = group_and_aggregate_data(df, selected_column, selected_agg)
            except Exception as e:
                st.error(f"Aggregation Error: {e}")
                st.stop()

            if selection_mode == "Party-wise":
                aggregated_df = aggregated_df.T

            st.write("### Grouped & Aggregated Data")
            st.dataframe(aggregated_df)

            # Perform PCA
            try:
                reduced_df = dimensionality_reduction(aggregated_df, num_components, [])
            except Exception as e:
                st.error(f"PCA Error: {e}")
                st.stop()

            display_pca_data(reduced_df, num_components)
        else:
            st.info("Configure your settings in the sidebar and click **Calculate & Display**.")


if __name__ == "__main__":
    main()
