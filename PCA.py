import numpy as np
import pandas as pd
from functions import load_data

# Example dataset with realistic categories
data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana"],  # Metadata (non-numeric)
    "JobRole": ["Engineer", "Scientist", "Engineer", "Manager"],  # Metadata (categorical)
    "Experience": [2, 5, 8, 10],  # Numerical data
    "Salary": [50000, 80000, 120000, 150000],  # Numerical data
    "Satisfaction": [3.5, 4.2, 2.8, 3.9]  # Numerical data
}


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    Perform dimensionality reduction (manual PCA) on a DataFrame.

    :param df: pd.DataFrame - The input DataFrame.
    :param num_components: int - Number of principal components to retain.
    :param meta_columns: list[str] - List of metadata columns that should not be transformed.
    :return: pd.DataFrame - DataFrame with the metadata and reduced components.
    """
    df.fillna(0, inplace=True)

    # Extract metadata columns and remove them from the DataFrame
    metadata = df[meta_columns]
    metadata_removed = df.drop(columns=meta_columns)

    # Fill missing values with column mean
    metadata_removed = metadata_removed.fillna(metadata_removed.mean())

    # Standardize data (zero mean)
    df_standardized = metadata_removed - metadata_removed.mean()

    # Convert standardized DataFrame to NumPy array
    standardized_array = df_standardized.to_numpy()

    # Calculate covariance matrix
    cov_matrix = np.cov(standardized_array, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indexes = np.argsort(eig_values)[::-1]
    sorted_eig_vectors = eig_vectors[:, sorted_indexes]

    # Select top eigenvectors according to the number of components
    top_eig_vectors = sorted_eig_vectors[:, :num_components]

    # Project the data into the lower-dimensional space
    reduced_data_array = standardized_array @ top_eig_vectors

    # Create new column names for principal components
    column_names = [f"PC{i + 1}" for i in range(num_components)]
    reduced_df = pd.DataFrame(reduced_data_array, columns=column_names, index=df_standardized.index)

    # Combine metadata with the reduced data
    final_df = pd.concat([metadata, reduced_df], axis=1)

    return final_df


if __name__ == '__main__':
    df_example = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df_example)

    meta_columns_list = ['Name', 'JobRole']

    # Apply manual PCA
    print("\nManual PCA:")
    manual_pca_df = dimensionality_reduction(df_example, 2, meta_columns_list)
    print(manual_pca_df)

    print("\nLoading Knesset data:")
    file_name = "knesset_25.xlsx"
    df_from_file = load_data(file_name)

    print("\nOriginal Knesset DataFrame:")
    print(df_from_file)

    meta_columns_list = ['city_name', 'ballot_code']

    # Apply manual PCA on Knesset data
    print("\nManual PCA on Knesset data:")
    manual_pca_df_from_file = dimensionality_reduction(df_from_file, 2, meta_columns_list)
    print(manual_pca_df_from_file)
