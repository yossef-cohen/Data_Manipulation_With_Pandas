import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a file (CSV or Excel) into a Pandas DataFrame.

    :param filepath: str - The file path of the CSV or Excel file to load.
    :return: pd.DataFrame - The loaded data as a Pandas DataFrame.

    Notes:
    - Supports files with extensions '.csv', '.xlsx', and '.xls'.
    - Prints an error message and exits if the file cannot be read or has an unsupported format.
    """
    try:
        if filepath[-3:] == 'csv':
            return pd.read_csv(filepath)
        elif filepath[-4:] == 'xlsx' or filepath[-3:] == 'xls':
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        exit(1)


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    """
    Group the DataFrame by a specified column and apply an aggregation function.

    :param df: pd.DataFrame - The input DataFrame.
    :param group_by_column: str - The column name to group by.
    :param agg_func: str or callable - The aggregation function (e.g., 'sum', 'mean', or a callable like np.sum).
    :return: pd.DataFrame - The grouped and aggregated DataFrame.
    """
    try:

        # Strip spaces in column names just in case
        df.columns = df.columns.str.strip()

        # Drop 'ballot_code' if it exists
        if 'ballot_code' in df.columns:
            df = df.drop(columns='ballot_code')  # Drop without errors

        # Convert callable to its name if needed
        agg_func = agg_func if isinstance(agg_func, str) else agg_func.__name__

        # Ensure there is a valid numeric column
        numeric_columns = df.select_dtypes(include='number').columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for aggregation.")

        # Perform the grouping and aggregation
        grouped = df[numeric_columns].groupby(df[group_by_column]).aggregate(agg_func)

        return grouped
    except Exception as e:
        print(f"Error in aggregation function: {e}")
        return pd.DataFrame()




def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove columns in the DataFrame where the sum of numeric values is below a specified threshold.

    :param df: pd.DataFrame - The input DataFrame.
    :param threshold: int - The minimum sum of values for a column to be kept.
    :return: pd.DataFrame - A DataFrame with sparse columns removed.
    """
    numeric_df = df.select_dtypes(include=["number"])  # Only include numeric columns
    return numeric_df.loc[:, numeric_df.sum() > threshold]
