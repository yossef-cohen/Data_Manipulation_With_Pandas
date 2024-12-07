import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PCA import dimensionality_reduction


def setup_page():
    st.set_page_config(
        page_title="Election Data Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("ניתוח נתוני בחירות")


def handle_file_upload():
    """Handle file upload and data loading"""
    st.header("1. העלאת נתונים")
    uploaded_file = st.file_uploader(
        "העלה קובץ נתונים",
        type=['csv', 'xlsx', 'xls'],
        help="תומך בקבצי CSV ו-Excel"
    )

    if uploaded_file is None:
        st.info("אנא העלה קובץ נתונים להתחלת הניתוח")
        return None

    try:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'city_name' not in df.columns:
            st.error("הקובץ חייב להכיל עמודה בשם 'city_name'")
            return None

        return df
    except Exception as e:
        st.error(f"שגיאה בטעינת הקובץ: {str(e)}")
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

        group_options = {
            'city_name': 'קיבוץ לפי עיר',
            'ballot_code': 'קיבוץ לפי קלפי'
        }
        group_by = st.selectbox(
            "בחר שדה לקיבוץ",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x]
        )

    with col2:
        agg_options = {
            "sum": "סכום קולות",
            "mean": "ממוצע קולות",
            "median": "חציון קולות"
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
        col_sums = aggregated.sum()
        significant_cols = col_sums[col_sums > threshold].index
        filtered_df = aggregated[significant_cols]

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


def visualize_pca_results(df, n_components=2):
    """Visualize PCA results with interpretation"""
    try:
        # Apply PCA using the original implementation
        meta_columns = []  # No metadata columns since we've already processed the data
        pca_result = dimensionality_reduction(df, n_components, meta_columns)

        # Create the visualization
        fig = px.scatter(
            pca_result,
            x='PC1',
            y='PC2',
            text=pca_result.index,
            title="מפת פיזור של נתוני הבחירות",
            labels={'PC1': 'רכיב ראשון', 'PC2': 'רכיב שני'}
        )

        # Update the layout
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

        # Interpret the components
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

    # Handle file upload
    df = handle_file_upload()
    if df is None:
        return

    # Show full data table
    st.subheader("הנתונים שנטענו")
    st.dataframe(df, use_container_width=True, height=400)

    # Get analysis parameters
    analysis_type, group_by, agg_func, n_components, threshold = get_analysis_params(df)

    if st.button("עבד נתונים", key="process_button"):
        # Process data using the temporary handler
        processed_df = process_data_for_pca(df, group_by, agg_func, threshold)

        if processed_df is None:
            st.error("שגיאה בעיבוד הנתונים")
            return

        # Show the shape of the processed data
        st.write(f"מספר שורות בטבלה המסוננת: {processed_df.shape[0]}")
        st.write(f"מספר עמודות בטבלה המסוננת: {processed_df.shape[1]}")

        # Transpose data if analyzing by party
        if analysis_type == "ניתוח לפי מפלגות":
            processed_df = processed_df.T
            st.write("(הנתונים הועברו - ניתוח לפי מפלגות)")

        # Visualize the results
        pca_result = visualize_pca_results(processed_df, n_components)

        if pca_result is not None:
            # Add download buttons
            col1, col2 = st.columns(2)

            with col1:
                processed_csv = processed_df.to_csv(index=True)
                st.download_button(
                    "הורד נתונים מעובדים (CSV)",
                    processed_csv,
                    "processed_election_data.csv",
                    "text/csv",
                    key='download-processed-csv'
                )

            with col2:
                pca_csv = pca_result.to_csv(index=True)
                st.download_button(
                    "הורד נתונים מופחתי ממדים (CSV)",
                    pca_csv,
                    "reduced_election_data.csv",
                    "text/csv",
                    key='download-reduced-csv'
                )


if __name__ == "__main__":
    main()