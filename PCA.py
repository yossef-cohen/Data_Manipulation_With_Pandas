import numpy as np
import pandas as pd
from sklearn.decomposition import PCA #imported this just to compare with our code
from DataFrameFromFile import load_data

# Example dataset with realistic categories
data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana"],         # Metadata (non-numeric)
    "JobRole": ["Engineer", "Scientist", "Engineer", "Manager"],  # Metadata (categorical)
    "Experience": [2, 5, 8, 10],                          # Numerical data
    "Salary": [50000, 80000, 120000, 150000],             # Numerical data
    "Satisfaction": [3.5, 4.2, 2.8, 3.9]                  # Numerical data
}

def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    # save the metadata and save it for later use (the metadata are the columns that don't impact the data)
    metadata = df[meta_columns]
    metadata_removed = df.drop(columns=meta_columns).dropna()

    # standardize the data (shift the data so that the mean is around zero and have a sum of 1)
    df_standardized = ((metadata_removed - metadata_removed.mean()) / metadata_removed.std())

    # Time to switch to an array since we're doing heavy math
    standardized_array = df_standardized.to_numpy()

    # find the covariance matrix (since we converted this from a data frame we need to use the rowvar=False
    cov_matrix = np.cov(standardized_array, rowvar=False)

    # find the eigenvalues and eigenvectors (this will tell me which column is pitching in the most to the variance)
    eig, eig_matrix = np.linalg.eig(cov_matrix)

    # find the sorted indexes from the large to small
    sorted_indexes = np.argsort(eig)[::-1]

    # sort by the index
    sorted_eig_matrix = eig_matrix[:, sorted_indexes]

    # select the top eigenvectors (columns) according to the argument num_components
    top_eig = sorted_eig_matrix[:, :num_components]

    # project the data into lower dimension plane according to the incorporate of the eigenvectors. (Don't forget @ is just matrix multiplication)
    # as far as I can tell we take the most important eigen vectors and according to the amount we take that's the amount of columns we will have in the end.
    reduced_data_array = standardized_array @ top_eig

    # create new column name
    columns_names = [f"PC{i+1}" for i in range(num_components)]

    reduced_df = pd.DataFrame(reduced_data_array, columns=columns_names, index=df_standardized.index)

    # combine the metadata with the reduced dataframe
    final_df = pd.concat([metadata, reduced_df], axis=1)

    return final_df

# this function fully from chatGPT just for testing purposes
def sklearn_pca(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    # Save metadata
    metadata = df[meta_columns]

    # Remove metadata
    metadata_removed = df.drop(columns=meta_columns).dropna()

    # Standardize the data
    df_standardized = ((metadata_removed - metadata_removed.mean()) / metadata_removed.std())

    # Apply PCA using scikit-learn
    pca = PCA(n_components=num_components)
    pca_components = pca.fit_transform(df_standardized)

    # Combine metadata with reduced data
    final_df = pd.concat([metadata, pd.DataFrame(pca_components, columns=[f"PC{i+1}" for i in range(num_components)])], axis=1)
    return final_df

# TODO so the reason this says that the answers are different is because of the sign of the eigenvectors which is fine since we dont care about size we care about direction. (should we change this or nah?)
if __name__ == '__main__':
    df_example = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df_example)

    meta_columns_list = ['Name', 'JobRole']

    # Apply manual PCA
    print("\nManual PCA:")
    manual_pca_df = dimensionality_reduction(df_example, 2, meta_columns_list)
    print(manual_pca_df)

    # Apply sklearn PCA
    print("\nscikit-learn PCA (for comparison):")
    sklearn_pca_df = sklearn_pca(df_example, 2, meta_columns_list)
    print(sklearn_pca_df)

    if manual_pca_df.equals(sklearn_pca_df):
        print("\nThe DataFrames are equal. Yay! we did it!")
    else:
        print("\nThe DataFrames are not equal. we suck!!")
        result = sklearn_pca_df == manual_pca_df
        print(result)

    print("\nAnd now we try the knesset data:")
    file_name = "knesset_25.xlsx"
    df_from_file = load_data(file_name)

    print("\nOriginal DataFrame:")
    print(df_from_file)

    meta_columns_list = ['city_name', 'ballot_code']

    # Apply manual PCA
    print("\nManual PCA:")
    manual_pca_df_from_file = dimensionality_reduction(df_from_file, 2, meta_columns_list)
    print(manual_pca_df_from_file)

    # Apply sklearn PCA
    print("\nscikit-learn PCA (for comparison):")
    sklearn_pca_df_from_file = sklearn_pca(df_from_file, 2, meta_columns_list)
    print(sklearn_pca_df_from_file)

    if sklearn_pca_df_from_file.equals(manual_pca_df_from_file):
        print("\nThe DataFrames are equal. Yay! we did it!")
    else:
        print("\nThe DataFrames are not equal. we suck!!")
        result = sklearn_pca_df_from_file == manual_pca_df_from_file
        print(result)