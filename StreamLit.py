import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from numpy.ma.core import nonzero

from PCA import dimensionality_reduction

AGG_OPTIONS = {
    "sum": "Sum of Votes",
    "mean": "Average Votes",
    "median": "Median Votes",
    "min": "Minimum Votes",
    "max": "Maximum Votes",
    "count": "Count of Entries",
    "std": "Standard Deviation of Votes",
    "var": "Variance of Votes",
    "first": "First Entry",
    "last": "Last Entry",
    "prod": "Product of Votes"
}


def setup_page():
    st.set_page_config(
        page_title="Election Data Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Election Data Analysis")


def handle_file_upload():
    """Handle file upload and data loading with error handling."""
    st.header("1. Upload Data")

    # File uploader supports CSV and Excel files
    uploaded_file = st.file_uploader(
        "Upload a data file",
        type=['csv', 'xlsx', 'xls'],
        help="Supports CSV and Excel files"
    )

    if uploaded_file is None:
        st.info("Please upload a data file to start the analysis")
        return None

    try:
        # Determine file type and load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        # Check for the required column
        if 'city_name' not in df.columns:
            st.error("The file must contain a 'city_name' column")
            return None

        # Successfully loaded data
        return df

    except pd.errors.ParserError:
        st.error("Error parsing the file. Ensure the CSV file is valid.")
        return None
    except ValueError:
        st.error("Error processing the file. Ensure the Excel file is valid.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def get_analysis_params(df):
    """Get analysis parameters from user input using a form."""
    st.header("2. Set Parameters")

    with st.form("parameter_form"):
        col1, col2 = st.columns(2)

        with col1:
            analysis_type = st.radio(
                "Analysis Type",
                ["Analysis by Cities", "Analysis by Parties"],
                help="Choose the desired analysis type"
            )

            group_by = st.selectbox(
                "Select Grouping Field",
                options=df.columns,
                help="Choose the field to group by"
            )

        with col2:
            agg_func = st.selectbox(
                "Aggregation Function",
                options=list(AGG_OPTIONS.keys()),
                format_func=lambda x: AGG_OPTIONS[x],
                help="Choose the aggregation function"
            )

            num_components = st.radio(
                "Number of Components",
                [2, 3],
                format_func=lambda x: f"{x}D View",
                help="Select the number of dimensions for visualization"
            )

        threshold = st.slider(
            "Minimum Vote Threshold",
            min_value=0,
            max_value=10000,
            value=1000,
            step=100,
            help="Filter cities/parties with fewer votes than the selected threshold"
        )

        # Form submission button
        submitted = st.form_submit_button("Submit")

    # Return parameters only if the form is submitted
    if submitted:
        return {
            "analysis_type": analysis_type,
            "group_by": group_by,
            "agg_func": agg_func,
            "num_components": num_components,
            "threshold": threshold
        }
    else:
        return None


def process_data_for_pca(df, group_by, agg_func, threshold):
    """Process data for PCA while preserving the original structure."""
    try:
        # Step 1: Group and aggregate data
        if group_by == 'city_name':
            if 'ballot_code' in df.columns:
                df = df.drop(columns='ballot_code')
            aggregated = df.groupby(group_by).agg(agg_func)
        else:
            aggregated = df.groupby(group_by).agg(agg_func)

        # Step 2: Handle missing and invalid values
        numeric_aggregated = aggregated.select_dtypes(include=['number']).fillna(0)

        # Replace extremely large values with a cap (e.g., 1e9)
        numeric_aggregated = numeric_aggregated.clip(upper=1e9)

        # Replace inf and NaN values
        numeric_aggregated.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_aggregated.dropna(axis=0, how='any', inplace=True)  # Drop rows with NaN
        numeric_aggregated.dropna(axis=1, how='any', inplace=True)  # Drop columns with NaN

        # Filter columns based on threshold
        col_sums = numeric_aggregated.sum()
        significant_cols = col_sums[col_sums > threshold].index
        filtered_df = numeric_aggregated[significant_cols]

        # Ensure enough columns for PCA
        if filtered_df.shape[1] < 2:
            st.error("Not enough valid data after filtering to perform PCA.")
            return None

        # Normalize data for PCA to avoid numerical instability
        filtered_df = (filtered_df - filtered_df.mean()) / filtered_df.std()

        # Debug information
        st.write("Data processing steps:")
        st.write(f"- Original shape: {df.shape}")
        st.write(f"- After aggregation: {aggregated.shape}")
        st.write(f"- After filtering: {filtered_df.shape}")

        # Display the filtered data table
        st.subheader("Filtered Data")
        st.dataframe(filtered_df, use_container_width=True, height=400, hide_index=False)

        return filtered_df

    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        return None


def interpret_pca_components(original_df, pca_result):
    """Interpret what each principal component represents"""
    st.subheader("Principal Component Analysis Interpretation")

    # Calculate correlations between original variables and PCs
    correlations = pd.DataFrame(index=original_df.columns)

    # Calculate correlations for PC1 and PC2
    for pc in ['PC1', 'PC2']:
        corr = [np.corrcoef(original_df[col], pca_result[pc])[0, 1] for col in original_df.columns]
        correlations[pc] = corr

    # Sort correlations by absolute value
    pc1_correlations = correlations['PC1'].abs().sort_values(ascending=False)
    pc2_correlations = correlations['PC2'].abs().sort_values(ascending=False)

    # Display top correlations for each PC
    col1, col2 = st.columns(2)

    with col1:
        st.write("Correlations with First Component (PC1):")
        for var in pc1_correlations.head(5).index:
            correlation = correlations.loc[var, 'PC1']
            direction = "Positive" if correlation > 0 else "Negative"
            st.write(f"{var}: {abs(correlation):.3f} ({direction} correlation)")

    with col2:
        st.write("Correlations with Second Component (PC2):")
        for var in pc2_correlations.head(5).index:
            correlation = correlations.loc[var, 'PC2']
            direction = "Positive" if correlation > 0 else "Negative"
            st.write(f"{var}: {abs(correlation):.3f} ({direction} correlation)")

    # Find extreme points
    st.subheader("Extreme Points")

    # Get top and bottom 3 points for each PC
    for pc in ['PC1', 'PC2']:
        st.write(f"Extreme Points for {pc}:")
        extreme_points = pca_result.sort_values(by=pc)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Low Values:")
            for idx in extreme_points.head(3).index:
                st.write(f"{idx}: {extreme_points.loc[idx, pc]:.2f}")

        with col2:
            st.write("High Values:")
            for idx in extreme_points.tail(3).index:
                st.write(f"{idx}: {extreme_points.loc[idx, pc]:.2f}")


def visualize_pca_results(df, n_components):
    """Visualize PCA results with interpretation"""
    try:
        # Apply PCA using the original implementation
        meta_columns = []  # No metadata columns since we've already processed the data
        pca_result = dimensionality_reduction(df, n_components, meta_columns)

        # Ensure complex numbers are converted to real if present
        if pca_result.select_dtypes(include=['complex']).size > 0:
            pca_result = pca_result.apply(lambda col: col.map(lambda x: x.real) if col.dtype == 'complex' else col)

        # Reverse text for proper RTL display
        def reverse_hebrew_text(text):
            return ''.join(reversed(text)) if any("\u0590" <= c <= "\u05FF" for c in text) else text

        pca_result['reversed_text'] = pca_result.index.map(reverse_hebrew_text)
        pca_result['original_text'] = pca_result.index  # Keep original text for tooltips

        # 3D Visualization if n_components is 3
        if n_components == 3:
            fig = px.scatter_3d(
                pca_result,
                x='PC1',
                y='PC2',
                z='PC3',
                text='reversed_text',  # Use reversed text for graph labels
                hover_name='original_text',  # Use original text for tooltips
                title="3D Scatter Plot of Election Data",
                labels={'PC1': 'First Component', 'PC2': 'Second Component', 'PC3': 'Third Component'}
            )
        else:
            # 2D Visualization for n_components = 2
            fig = px.scatter(
                pca_result,
                x='PC1',
                y='PC2',
                text='reversed_text',  # Use reversed text for graph labels
                hover_name='original_text',  # Use original text for tooltips
                title="2D Scatter Plot of Election Data",
                labels={'PC1': 'First Component', 'PC2': 'Second Component'},
            )

        # Update layout and marker settings
        fig.update_traces(
            textposition='top center',
            marker=dict(size=12, opacity=0.7, color='blue'),  # Set marker color to black
            mode='markers+text',
            textfont=dict(color='red', family='Arial')  # Set text color to black and ensure proper font
        )
        fig.update_layout(
            height=700,
            template='plotly_white',
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Interpret the components only for 2D PCA
        if n_components == 2:
            interpret_pca_components(df, pca_result)

        return pca_result

    except Exception as e:
        st.error(f"Error in visualization: {str(e)}")
        st.write("Debug information:")
        st.write("Input data shape:", df.shape)
        st.write("Input data columns:", df.columns.tolist())
        return None


def main():
    setup_page()

    # Handle file upload and save data in session state
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None

    df = handle_file_upload()
    if df is not None:
        st.session_state.uploaded_data = df

    # Ensure data is loaded before proceeding
    if st.session_state.uploaded_data is None:
        return

    # Display loaded data
    st.subheader("Loaded Data")
    st.dataframe(st.session_state.uploaded_data, use_container_width=True, height=400)

    # Get analysis parameters
    params = get_analysis_params(st.session_state.uploaded_data)

    if params:
        # Process data and visualize results only after parameters are submitted
        processed_df = process_data_for_pca(
            st.session_state.uploaded_data,
            params["group_by"],
            params["agg_func"],
            params["threshold"]
        )

        if processed_df is not None:
            # Transpose data if analyzing by party
            if params["analysis_type"] == "Analysis by Parties":
                processed_df = processed_df.T
                st.write("(Data Transposed - Analysis by Parties)")

            # Visualize the results
            visualize_pca_results(processed_df, params["num_components"])


if __name__ == "__main__":
    main()
