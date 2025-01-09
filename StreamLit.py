import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from numpy.ma.core import nonzero

from PCA import dimensionality_reduction


def setup_page():
    st.set_page_config(
        page_title="Election Data Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("ניתוח נתוני בחירות")


def handle_file_upload():
    """Handle file upload and data loading with error handling."""
    st.header("1. העלאת נתונים")

    # File uploader supports CSV and Excel files
    uploaded_file = st.file_uploader(
        "העלה קובץ נתונים",
        type=['csv', 'xlsx', 'xls'],
        help="תומך בקבצי CSV ו-Excel"
    )

    if uploaded_file is None:
        st.info("אנא העלה קובץ נתונים להתחלת הניתוח")
        return None

    try:
        # Determine file type and load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Check for the required column
        if 'city_name' not in df.columns:
            st.error("הקובץ חייב להכיל עמודה בשם 'city_name'")
            return None

        # Successfully loaded data
        return df

    except pd.errors.ParserError:
        st.error("שגיאה בניתוח הקובץ. ודא שקובץ ה-CSV תקין.")
        return None
    except ValueError:
        st.error("שגיאה בעיבוד הקובץ. ודא שקובץ ה-Excel תקין.")
        return None
    except Exception as e:
        st.error(f"שגיאה בלתי צפויה: {str(e)}")
        return None


def get_analysis_params(df):
    """Get analysis parameters from user input"""
    st.header("2. הגדרת פרמטרים")

    col1, col2 = st.columns(2)

    with col1:
        analysis_type = st.radio(
            "סוג ניתוח",
            ["ניתוח לפי ערים", "ניתוח לפי מפלגות"],
            help="בחר את אופן הניתוח הרצוי"
        )

        group_options = {col: f"{col}" for col in df.columns}

        group_by = st.selectbox(
            "בחר שדה לקיבוץ",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x]
        )

    with col2:
        agg_options = {
            "sum": "סכום קולות",
            "mean": "ממוצע קולות",
            "median": "חציון קולות",
            "min": "Minimum"
        }
        agg_func = st.selectbox(
            "פונקציית אגרגציה",
            options=list(agg_options.keys()),
            format_func=lambda x: agg_options[x]
        )

        num_components = st.radio(
            "מספר רכיבים",
            [2, 3],
            format_func=lambda x: f"{x}D תצוגה",
            help="בחר מספר ממדים להצגת התוצאות"
        )

    threshold = st.slider(
        "סף מינימום לקולות",
        min_value=0,
        max_value=10000,
        value=1000,
        step=100,
        help="סינון מפלגות/ערים עם פחות קולות מהסף שנבחר"
    )

    return analysis_type, group_by, agg_func, num_components, threshold


def process_data_for_pca(df, group_by, agg_func, threshold):
    """Process data for PCA while preserving the original structure"""
    try:
        # Step 1: Group and aggregate data
        if group_by == 'city_name':
            aggregated = df.drop(columns='ballot_code').groupby(group_by).agg(agg_func)
        else:
            aggregated = df.groupby(group_by).agg(agg_func)

        # Step 2: Remove sparse columns
        numeric_aggregated = aggregated.select_dtypes(include=['number'])  # Keep only numeric columns
        col_sums = numeric_aggregated.sum()
        significant_cols = col_sums[col_sums > threshold].index
        filtered_df = numeric_aggregated[significant_cols]

        # Debug information
        st.write("Data processing steps:")
        st.write(f"- Original shape: {df.shape}")
        st.write(f"- After aggregation: {aggregated.shape}")
        st.write(f"- After filtering: {filtered_df.shape}")

        # Display the filtered data table
        st.subheader("נתונים מסוננים")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            hide_index=False
        )

        return filtered_df

    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        return None


def interpret_pca_components(original_df, pca_result):
    """Interpret what each principal component represents"""
    st.subheader("ניתוח משמעות הרכיבים העיקריים")

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
        st.write("מתאמים עם רכיב ראשון (PC1):")
        for var in pc1_correlations.head(5).index:
            correlation = correlations.loc[var, 'PC1']
            direction = "חיובי" if correlation > 0 else "שלילי"
            st.write(f"{var}: {abs(correlation):.3f} (מתאם {direction})")

    with col2:
        st.write("מתאמים עם רכיב שני (PC2):")
        for var in pc2_correlations.head(5).index:
            correlation = correlations.loc[var, 'PC2']
            direction = "חיובי" if correlation > 0 else "שלילי"
            st.write(f"{var}: {abs(correlation):.3f} (מתאם {direction})")

    # Find extreme points
    st.subheader("נקודות קיצון")

    # Get top and bottom 3 points for each PC
    for pc in ['PC1', 'PC2']:
        st.write(f"נקודות קיצון עבור {pc}:")
        extreme_points = pca_result.sort_values(by=pc)

        col1, col2 = st.columns(2)

        with col1:
            st.write("ערכים נמוכים:")
            for idx in extreme_points.head(3).index:
                st.write(f"{idx}: {extreme_points.loc[idx, pc]:.2f}")

        with col2:
            st.write("ערכים גבוהים:")
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

        # 3D Visualization if n_components is 3
        if n_components == 3:
            fig = px.scatter_3d(
                pca_result,
                x='PC1',
                y='PC2',
                z='PC3',
                text=pca_result.index,
                title="מפת פיזור תלת-ממדית של נתוני הבחירות",
                labels={'PC1': 'רכיב ראשון', 'PC2': 'רכיב שני', 'PC3': 'רכיב שלישי'}
            )
        else:
            # 2D Visualization for n_components = 2
            fig = px.scatter(
                pca_result,
                x='PC1',
                y='PC2',
                text=pca_result.index,
                title="מפת פיזור של נתוני הבחירות",
                labels={'PC1': 'רכיב ראשון', 'PC2': 'רכיב שני'}
            )

        # Update layout and marker settings
        fig.update_traces(
            textposition='top center',
            marker=dict(size=12, opacity=0.7),
            mode='markers+text'
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
    st.subheader("הנתונים שנטענו")
    st.dataframe(st.session_state.uploaded_data, use_container_width=True, height=400)

    # Collect parameters and store them in session state
    st.header("2. הגדרת פרמטרים")

    if "params" not in st.session_state:
        st.session_state.params = {
            "analysis_type": "ניתוח לפי ערים",
            "group_by": "city_name",
            "agg_func": "sum",
            "n_components": 2,
            "threshold": 1000,
        }

    # Input widgets for analysis parameters
    with st.form("parameter_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params["analysis_type"] = st.radio(
                "סוג ניתוח",
                ["ניתוח לפי ערים", "ניתוח לפי מפלגות"],
                help="בחר את אופן הניתוח הרצוי"
            )
            st.session_state.params["group_by"] = st.selectbox(
                "בחר שדה לקיבוץ",
                options=st.session_state.uploaded_data.columns
            )

        with col2:
            st.session_state.params["agg_func"] = st.selectbox(
                "פונקציית אגרגציה",
                ["sum", "mean", "median", "min"],
                format_func=lambda x: {"sum": "סכום קולות", "mean": "ממוצע קולות", "median": "חציון קולות", "min": "Minimum"}[x]
            )
            st.session_state.params["n_components"] = st.radio(
                "מספר רכיבים",
                [2, 3],
                format_func=lambda x: f"{x}D תצוגה"
            )

        st.session_state.params["threshold"] = st.slider(
            "סף מינימום לקולות",
            min_value=0,
            max_value=10000,
            value=1000,
            step=100,
            help="סינון מפלגות/ערים עם פחות קולות מהסף שנבחר"
        )

        # Submit button for the form
        submitted = st.form_submit_button("עבד נתונים")

    # Process data and visualize results only after form submission
    if submitted:
        processed_df = process_data_for_pca(
            st.session_state.uploaded_data,
            st.session_state.params["group_by"],
            st.session_state.params["agg_func"],
            st.session_state.params["threshold"]
        )

        if processed_df is not None:
            # Transpose data if analyzing by party
            if st.session_state.params["analysis_type"] == "ניתוח לפי מפלגות":
                processed_df = processed_df.T
                st.write("(הנתונים הועברו - ניתוח לפי מפלגות)")

            # Visualize the results
            visualize_pca_results(processed_df, st.session_state.params["n_components"])



if __name__ == "__main__":
    main()