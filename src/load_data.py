import pandas as pd

# loads the csv into a pandas dataframe
def load_data(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path,compression='gzip')
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path} returning empty df")

        return pd.DataFrame()  # Return an empty DataFrame if the file is not found