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
    df_sub_mean = df - df.mean()
    df_standardized = df_sub_mean / df.std()
    pass


print((group_and_aggregate_data(load_data('knesset_25.xlsx'), 'city_name', min)))
