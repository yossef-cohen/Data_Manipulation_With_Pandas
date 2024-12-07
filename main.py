import pandas as pd
import numpy as np
from DataFrameFromFile import load_data
from matplotlib import pylab as plt
from PCA import dimensionality_reduction


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    try:
        agg_func = agg_func if isinstance(agg_func, str) else agg_func.__name__
        return df.drop(columns='ballot_code').groupby(group_by_column).aggregate(agg_func)
    except Exception as e:
        print(f"not a aggregate function: {e}")
        return pd.DataFrame()


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return df[df.sum()[df.sum() > threshold].index]


aggregated_df = group_and_aggregate_data(load_data("knesset_25.xlsx"), "city_name", "sum")
removed_df = remove_sparse_columns(aggregated_df, 1000)
dr_df = dimensionality_reduction(removed_df, 2, ['party_avoda'])

labels = list(range(len(dr_df.index)))
plt.figure(figsize=(6, 6))
for cluster in np.unique(labels):
    cluster_data = dr_df[labels == cluster]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f"Cluster {cluster}")

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Reduced data")
plt.legend()
plt.show()