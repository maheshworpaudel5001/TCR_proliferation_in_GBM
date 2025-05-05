import numpy as np
import pandas as pd


def ccf(data):
    """
    Calculate CCF from a counts dictionary, list of vectors, or dataframe.

    Parameters:
    -----------
    data : dict, list, 1D-array or DataFrame
        The input data can be a dictionary with clone sizes as keys and counts as values,
        a list of clone sizes, or a 1D-array of clone sizes or a dataframe with the first column which has clone sizes and second column which has counts.

    Returns:
    --------
    clone sizes, ccf
    """
    if isinstance(data, dict):
        # Convert dictionary to dataframe
        counts_df = pd.DataFrame(list(data.items()), columns=["Clone Size", "Counts"])
    elif isinstance(data, list):
        # Convert list to counts dictionary and then to dataframe
        counts = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1
        counts_df = pd.DataFrame(list(counts.items()), columns=["Clone Size", "Counts"])
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        # Convert 1D-array to counts dictionary and then to dataframe
        counts = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1
        counts_df = pd.DataFrame(list(counts.items()), columns=["Clone Size", "Counts"])
    elif isinstance(data, pd.DataFrame):
        counts_df = data
        counts_df.columns = ["Clone Size", "Counts"]
    else:
        raise ValueError(
            "Data should be either a dictionary, list, 1D-array or dataframe. If dataframe, first column must have clone sizes and second column must have counts."
        )

    # Convert Clone Size and Counts to integers
    counts_df["Clone Size"] = counts_df["Clone Size"].astype(int)
    counts_df["Counts"] = counts_df["Counts"].astype(int)

    # Sort the dataframe by Clone Size
    counts_df = counts_df.sort_values(by="Clone Size")

    # Calculate CCF
    x, y0 = counts_df.iloc[:, 0].values, counts_df.iloc[:, 1].values
    y = (np.cumsum(y0[::-1])[::-1]) / sum(y0)

    return x.astype(int), y


if __name__ == "__main__":

    # Sample data as list and dictionary
    data_list = [1, 1, 1, 2, 2, 3, 5, 5, 5]
    data_dict = {"1": 3, "2": 2, "3": 1, "5": 3, "10": 2}

    # Calculate CCF for both data formats
    clone_sizes_list, ccf_list = ccf(data_list)
    clone_sizes_dict, ccf_dict = ccf(data_dict)

    # Print the results
    print("CCF from list data:")
    print(clone_sizes_list)
    print(ccf_list)

    print("\nCCF from dictionary data:")
    print(clone_sizes_dict)
    print(ccf_dict)
