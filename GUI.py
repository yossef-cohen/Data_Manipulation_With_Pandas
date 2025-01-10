import streamlit as st
import pandas as pd
import plotly.express as px

# Local imports - ensure these exist in your project
from functions import group_and_aggregate_data, remove_sparse_columns
from PCA import dimensionality_reduction


def display_pca_data(
    df: pd.DataFrame,
    num_components: int,
    label_col: str = None
) -> None:
    """
    Displays PCA-reduced data in 1D, 2D, or 3D using Plotly and Streamlit,
    with an optional label column for hover information.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'PC1', 'PC2', 'PC3', etc., plus optional label_col.
    num_components : int
        Number of principal components to visualize (1, 2, or 3 recommended).
    label_col : str, optional
        Name of the column to show on hover (e.g., "city_name" or "party").
    """

    # Build a hover_data dictionary to hide or show columns in the tooltip
    hover_data = {}
    for pc in ['PC1', 'PC2', 'PC3']:
        if pc in df.columns:
            hover_data[pc] = False  # Hide numeric PC values from hover

    if label_col and label_col in df.columns:
        hover_data[label_col] = True  # Show the label column on hover

    # Plot according to the number of components
    if num_components == 1:
        # Simulate 1D by plotting PC1 on x-axis and 0 on y-axis
        fig = px.scatter(
            df,
            x='PC1',
            y=[0]*len(df),
            title="1D PCA Visualization",
            hover_data=hover_data,
            labels={'PC1': 'X - PC1'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif num_components == 2:
        # 2D scatter plot
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            title="2D PCA Visualization",
            hover_data=hover_data,
            labels={'PC1': 'X - PC1', 'PC2': 'Y - PC2'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif num_components == 3:
        # 3D scatter plot
        fig = px.scatter_3d(
            df,
            x='PC1',
            y='PC2',
            z='PC3',
            title="3D PCA Visualization",
            hover_data=hover_data,
            labels={'PC1': 'X - PC1', 'PC2': 'Y - PC2', 'PC3': 'Z - PC3'}
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # If not in [1, 2, 3], show a warning and just display the DataFrame
        st.warning(
            "Number of components is outside the recommended range [1, 3]. "
            "Below is the raw PCA DataFrame instead."
        )

    st.write("### PCA Output DataFrame")
    st.dataframe(df)


def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Dimensionality Reduction", layout="wide")

    with st.sidebar:
        st.title("Data & Settings")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        df = None  # Will store the uploaded DataFrame

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to load the file. Error: {e}")
                st.stop()

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

                # Validate num_components
                try:
                    num_components = int(num_components_input)
                    if num_components < 1:
                        st.warning("Number of components must be at least 1. Setting to 1.")
                        num_components = 1
                except ValueError:
                    st.warning("Invalid number. Defaulting to 2D PCA.")
                    num_components = 2

                # Button to run
                valid_column = (selected_column != "(No Selection)")
                valid_agg = (selected_agg != "(No Selection)")
                run_button = st.button(
                    "Calculate & Display",
                    disabled=not (valid_column and valid_agg)
                )

    st.title("Dimensionality Reduction App")

    # If no file is uploaded or df is empty
    if df is None or df.empty:
        st.info("Please upload a valid CSV or Excel file in the sidebar to begin.")
        return

    # Create tabs for data preview and PCA visualization
    tab1, tab2 = st.tabs(["Data Preview", "PCA Visualization"])

    with tab1:
        st.subheader("1. Preview Original Data")
        st.dataframe(df)

    with tab2:
        st.subheader("2. PCA Results & Visualization")

        # Only proceed if user clicked the button
        if run_button:
            # 1) Group & aggregate
            try:
                aggregated_df = group_and_aggregate_data(df, selected_column, selected_agg)
                aggregated_df = remove_sparse_columns(aggregated_df, threshold=1000)
            except Exception as e:
                st.error(f"Aggregation Error: {e}")
                st.stop()

            # 2) City-wise vs Party-wise
            if selection_mode == "Party-wise":
                # Transpose so that rows become columns (parties)
                aggregated_df = aggregated_df.T
                # Rename index to 'party'
                aggregated_df = aggregated_df.reset_index().rename(columns={'index': 'party'})
                label_col = 'party'
            else:
                # Keep the grouping column in place; rename index to keep it as a column
                aggregated_df = aggregated_df.reset_index()
                label_col = selected_column

            st.write("### Grouped & Aggregated Data")
            st.dataframe(aggregated_df)

            # 3) Perform PCA
            # Include label_col so it remains in the DataFrame
            try:
                reduced_df = dimensionality_reduction(aggregated_df, num_components, [label_col])
            except Exception as e:
                st.error(f"Cannot do PCA on such Data!")
                st.stop()

            # 4) Display the PCA result, with custom hover labeling
            display_pca_data(reduced_df, num_components, label_col=label_col)

        else:
            st.info("Configure your settings in the sidebar and click **Calculate & Display**.")


if __name__ == "__main__":
    main()
