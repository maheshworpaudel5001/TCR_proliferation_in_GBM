import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def pdf(data):
    """
    Calculate pdf from a counts dictionary, list of vectors, or dataframe.

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
    y = y0 / sum(y0)
    return x.astype(int), y


def generate_configuration_per_tcr(
    prob_array,
    min_clone_size=1,
    max_clone_size=None,
    sample_size=1,
    disable_progressbar=False,
):
    if prob_array.ndim == 1:
        assert max_clone_size == min_clone_size + len(
            prob_array
        ), f"max_clone_size ({max_clone_size}) must equal min_clone_size ({min_clone_size}) + len(prob_array) ({len(prob_array)})"
        config_per_tcr = np.random.choice(
            range(min_clone_size, len(prob_array) + min_clone_size),
            size=sample_size,
            replace=True,
            p=prob_array / sum(prob_array),
        )
        return config_per_tcr
    elif prob_array.ndim == 2:  # Ensure we only handle 1D and 2D arrays
        assert (
            max_clone_size == min_clone_size + prob_array.shape[1]
        ), f"max_clone_size ({max_clone_size}) must equal min_clone_size ({min_clone_size}) + prob_array.shape[1] ({prob_array.shape[1]})"

        all_configs = np.empty((prob_array.shape[0], sample_size))
        for i in tqdm.tqdm(
            range(prob_array.shape[0]),
            total=prob_array.shape[0],
            desc="Generating configuration for each TCR",
            leave=False,
            disable=disable_progressbar,
        ):
            # config_per_tcr = generate_configuration_per_tcr(prob_array[i, :], size)
            config_per_tcr = np.random.choice(
                range(min_clone_size, prob_array.shape[1] + min_clone_size),
                size=sample_size,
                replace=True,
                p=prob_array[i, :] / sum(prob_array[i, :]),
            )
            all_configs[i, :] = config_per_tcr
        return all_configs
    else:
        raise ValueError("Input array must be either 1D or 2D.")


def process_column(col_data):
    values, probabilities = pdf(col_data)
    return values, probabilities


def ci_pdf(configs, alpha=5, least_number_of_observations=1):

    # configs must be a two dimensional array of shape (#tcr, #configs per tcr)
    if configs.ndim >= 2:
        pdf_dict = {}

        # Use ProcessPoolExecutor to parallelize the processing of columns
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    process_column, [configs[:, col] for col in range(configs.shape[1])]
                )
            )

        # Iterate over the results and populate the pdf_dict
        for values, probabilities in results:
            for value, probability in zip(values, probabilities):
                if value not in pdf_dict:
                    pdf_dict[value] = []
                pdf_dict[value].append(probability)

        # Calculate alpha% level of significance
        confidence_intervals = {}
        for key, probs in pdf_dict.items():
            if len(probs) > least_number_of_observations:
                median_prob = np.median(probs)
                lb = np.percentile(probs, alpha / 2)
                ub = np.percentile(probs, 100 - alpha / 2)
                confidence_intervals[key] = (lb, median_prob, ub)
            else:
                continue

        df = pd.DataFrame.from_dict(
            confidence_intervals, orient="index", columns=["lower", "median", "upper"]
        )

        # Add the keys as a column named 'clone_size'
        df.reset_index(inplace=True)
        df.rename(columns={"index": "clone_size"}, inplace=True)

        # Order the rows in increasing clone size
        df.sort_values(by="clone_size", inplace=True)

        if df.empty:
            raise ValueError("The resulting DataFrame is empty.")
        else:
            return df
    else:
        raise ValueError("CI cannot be calculated for single configuration.")


def plot_pdf(config_data, raw=True):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 5),
    )
    if raw:
        data = np.unique(config_data, return_counts=True)
        x_data, y_data = data[0], data[1] / sum(data[1])

    else:
        x_data, y_data = config_data[0], config_data[1]

    ax.scatter(
        x_data,
        y_data,
        s=50,
        facecolors="white",
        edgecolors="black",
        label="data",
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    offset = 0.30
    ax.set_xlim(min(x_data) - offset * min(x_data), max(x_data) + offset * max(x_data))
    ax.set_ylim(min(y_data) - offset * min(y_data), max(y_data) + offset * max(y_data))

    ax.legend(loc="best")
    return fig, ax


if __name__ == "__main__":

    # Sample data as list and dictionary
    data_list = [1, 1, 1, 2, 2, 3, 5, 5, 5, 5]
    data_dict = {"1": 3, "2": 2, "3": 1, "5": 4}

    # Calculate CCF for both data formats
    clone_sizes_list, pdf_list = pdf(data_list)
    clone_sizes_dict, pdf_dict = pdf(data_dict)

    # Print the results
    print("PDF from list data:")
    print(clone_sizes_list)
    print(pdf_list)

    print("\nPDF from dictionary data:")
    print(clone_sizes_dict)
    print(pdf_dict)
