import pandas as pd
import numpy as np
import streamlit as st


def load_data(filepath: str) -> pd.DataFrame:
    try :
        if filepath[-3:] == 'csv':
            return pd.read_csv(filepath)
        elif filepath[-4:] == 'xlsx' or filepath[-3:] == 'xls':
            return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        exit(1)


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    try:
        agg_func = agg_func if isinstance(agg_func, str) else agg_func.__name__
        return df.drop(columns='ballot_code').groupby(group_by_column).aggregate(agg_func)
    except Exception as e:
        print(f"not a aggregate function: {e}")
        return pd.DataFrame()


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return  df[df.sum()[df.sum() > threshold].index]


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    # save the metadata and save it for later use
    metadata = df[meta_columns]

    # remove the metadat for the df
    metadata_removed = df.drop(columns=meta_columns)

    # standardize the data shift the data around zero
    df_standardize = (metadata_removed - metadata.mean()) / metadata.std()

    # center the data around 0 and find the covariance
    cov_matrix = np.cov(df_standardize.T)

    # find the eigenvalues and eigenvectors
    eig, eig_matrix = np.linalg.eig(cov_matrix)

    # find the sorted indexes from the large to small
    sorted_indexes = np.argsort(eig)[::-1]

    # sort by the index
    sorted_eig_matrix = eig_matrix[:, sorted_indexes]

    # select the top eigenvectors (columns) according to the argument num_components
    top_eig = sorted_eig_matrix[:, :num_components]

    # project the data into lower dimension plane according to the incorporate of the eigenvectors
    reduced_data = df_standardize.dot(top_eig)

    # columns name
    columns_names = [f"PC{i+1}" for i in range(num_components)]

    # rename the columns to pc1, pc2 ...
    reduced_dg = pd.DataFrame(reduced_data, columns=columns_names)

    # combine the metadata with the reduced dataframe
    final_df = pd.concat([metadata_removed.reset_index(drop=True), reduced_dg], axis=1)

    return final_df


print((group_and_aggregate_data(load_data('knesset_25.xlsx'), 'city_name', min)))
