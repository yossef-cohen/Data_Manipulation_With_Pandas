import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    try :
        if filepath[-3:] == 'csv':
            return pd.read_csv(filepath)
        elif filepath[-4:] == 'xlsx' or filepath[-3:] == 'xls':
            return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        exit(1)