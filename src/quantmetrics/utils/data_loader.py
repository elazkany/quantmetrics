# utils/data_loader.py
import importlib.resources as pkg_resources
import pandas as pd

def load_data(file_name: str) -> pd.DataFrame:
    """
    Load a Parquet data file from the package's data directory.

    Parameters
    ----------
    file_name : str
        The name of the Parquet file to load (without the .parquet extension).

    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.
    """
    # Append ".parquet" to the file name
    full_file_name = f"{file_name}.parquet"
    
    # Use the new `files` method to access the data file
    data_path = pkg_resources.files('quantmetrics.data') / full_file_name
    with data_path.open('rb') as f:
        data = pd.read_parquet(f)
    return data
