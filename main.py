import pandas as pd
from DataFrameFromFile import load_data


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    try:
        agg_func = agg_func if isinstance(agg_func, str) else agg_func.__name__
        return df.drop(columns='ballot_code').groupby(group_by_column).aggregate(agg_func)
    except Exception as e:
        print(f"not a aggregate function: {e}")
        return pd.DataFrame()


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return  df[df.sum()[df.sum() > threshold].index]



print((group_and_aggregate_data(load_data('knesset_25.xlsx'), 'city_name', min)))
