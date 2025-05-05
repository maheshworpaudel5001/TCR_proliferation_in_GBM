import sys, os, glob

root_dir = "/home/gddaslab/mxp140/tcr_project_ultimate"
sys.path.append(root_dir)

import re
import argparse
import numpy as np
import pandas as pd
from probability import probability
from scipy.optimize import minimize
from constants import max_kr_panpep, max_kr_vdjdb, max_kr_mcpas

probability = np.vectorize(probability)

# from concurrent.futures import ProcessPoolExecutor


# def neg_likelihood(params, clone_count_values, scaled_kr_values, verbose=False):
#     x1, x2 = params
#     n = len(clone_count_values)
#     x1_values = np.full(n, x1)
#     x2_values = np.full(n, x2)
#     # Use ProcessPoolExecutor for better performance
#     with ProcessPoolExecutor() as executor:
#         probs = list(
#             executor.map(
#                 probability, x1_values, x2_values, scaled_kr_values, clone_count_values
#             )
#         )
#     # Replace zero values with the smallest positive value allowed in Python
#     smallest_positive_value = np.finfo(float).eps
#     probabilities = np.where(probs == 0, smallest_positive_value, probs)
#     sum_log_probs = np.sum(np.log(probabilities))
#     neg_sum = -sum_log_probs

#     if verbose:
#         print(f"Neg-logL: {neg_sum:.8f}")
#         print(f"x1: {x1:.8f}")
#         print(f"x2: {x2:.8f}")
#         print(f"=" * 80)
#     return neg_sum


# def new_neg_likelihood(
#     params, fixed_x1, clone_count_values, scaled_kr_values, verbose=False
# ):
#     # params is now just x2, fixed_x1 is provided separately
#     x2 = params[0]  # params is now a 1D array with just x2
#     x1 = fixed_x1
#     n = len(clone_count_values)
#     x1_values = np.full(n, x1)
#     x2_values = np.full(n, x2)
#     probs = probability(x1_values, x2_values, scaled_kr_values, clone_count_values)
#     # Replace zero values with the smallest positive value allowed in Python
#     smallest_positive_value = np.finfo(float).eps
#     probabilities = np.where(probs == 0, smallest_positive_value, probs)
#     sum_log_probs = np.sum(np.log(probabilities))
#     neg_sum = -sum_log_probs

#     if verbose:
#         print(f"Neg-logL: {neg_sum:.8f}")
#         print(f"x2: {x2:.8f}")
#         print(f"=" * 80)
#     return neg_sum


def new_neg_likelihood(
    params, fixed_xw, clone_count_values, scaled_kr_values, verbose=False
):
    # params is now just x2, fixed_x1 is provided separately
    xp = params[0]  # params is now a 1D array with just x2
    xw = fixed_xw
    n = len(clone_count_values)
    xp_values = np.full(n, xp)
    xw_values = np.full(n, xw)

    probs = probability(xw_values, xp_values, scaled_kr_values, clone_count_values)
    # Replace zero values with the smallest positive value allowed in Python
    smallest_positive_value = np.finfo(float).eps
    probabilities = np.where(probs == 0, smallest_positive_value, probs)
    sum_log_probs = np.sum(np.log(probabilities))
    neg_sum = -sum_log_probs

    if verbose:
        print(f"Neg-logL: {neg_sum:.8f}")
        print(f"xp: {xp:.8f}")
        print(f"=" * 80)
    return neg_sum


# def run_optimization(
#     clone_count_values,
#     scaled_kr_values,
#     bounds=(
#         (1e-10, 100),
#         (1e-10, 100),
#     ),
#     initial_guess=[5, 5],
#     verbose=False,
# ):
#     bounds = bounds
#     initial_guess = initial_guess
#     # initial_guess = [np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1])]
#     result = minimize(
#         neg_likelihood,
#         initial_guess,
#         args=(clone_count_values, scaled_kr_values, verbose),
#         method="Nelder-Mead",
#         bounds=bounds,
#     )

#     x1, x2 = result.x
#     return x1, x2, initial_guess[0], initial_guess[1], result.fun


def run_optimization(
    clone_count_values,
    scaled_kr_values,
    fixed_x1,
    bounds=((1e-10, 100),),
    initial_guess=[5],
    verbose=False,
):
    # Modified to take fixed_x1 parameter and only optimize x2
    bounds = [bounds[0]]  # Only bounds for x2
    initial_guess = initial_guess
    result = minimize(
        new_neg_likelihood,
        initial_guess,
        args=(fixed_x1, clone_count_values, scaled_kr_values, verbose),
        method="Nelder-Mead",
        bounds=bounds,
    )
    x2 = result.x[0]
    return fixed_x1, x2, initial_guess[0], result.fun


if __name__ == "__main__":
    # Add argument parsing for patient_id
    parser = argparse.ArgumentParser(description="Process patient ID.")
    parser.add_argument(
        "--patient_id", type=str, help="The ID of the patient to process"
    )
    parser.add_argument(
        "--dataset", type=str, help="Choose one from panpep or vdjdb or mcpas"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Set to True if you want to minimize by regions",
    )
    parser.add_argument(
        "--region_num",
        type=str,
        help='If region is true provide region number. If you want to do for all regions at once provide input "all".',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during optimization",
    )  # Added verbose argument
    args = parser.parse_args()

    patient_id = args.patient_id
    dataset = args.dataset
    region = args.region
    region_num = args.region_num
    verbose = args.verbose
    print(f"Processing data for patient ID: {patient_id}-{dataset}")

    if dataset == "panpep":
        full_data = pd.read_csv(
            f"{root_dir}/data/BrMET_and_GBM_data-PANPEP.csv", sep=","
        )
        max_kr = max_kr_panpep
    elif dataset == "vdjdb":
        full_data = pd.read_csv(
            f"{root_dir}/data/BrMET_and_GBM_data-ERGOII.csv",
            sep=",",
            usecols=["Patient", "CDR3", "counts", "kr_vdjdb"],
        )
        full_data.rename(columns={"kr_vdjdb": "kr"}, inplace=True)
        max_kr = max_kr_vdjdb
    elif dataset == "mcpas":
        full_data = pd.read_csv(
            f"{root_dir}/data/BrMET_and_GBM_data-ERGOII.csv",
            sep=",",
            usecols=["Patient", "CDR3", "counts", "kr_mcpas"],
        )
        full_data.rename(columns={"kr_mcpas": "kr"}, inplace=True)
        max_kr = max_kr_mcpas
    else:
        raise FileNotFoundError(f"Data not found.")

    fixed_x1 = 100
    # Load the results file to upload results
    results_file = f"{root_dir}/results/results.xlsx"
    params_df = pd.read_excel(results_file, sheet_name="parameters", engine="openpyxl")

    if region == "False":
        patient_data = full_data[full_data["Patient"] == patient_id]
        if patient_data.empty:
            print(f"No data available for patient ID: {patient_id}. Skipping...")
            sys.exit(1)
        clone_count_values = patient_data["counts"].values
        kr_values = patient_data["kr"].values
        scaled_kr_values = kr_values / max_kr
        result = run_optimization(
            clone_count_values=clone_count_values,
            scaled_kr_values=scaled_kr_values,
            fixed_x1=fixed_x1,  # np.full(len(scaled_kr_values), fixed_x1),
            verbose=verbose,
        )
        x1, x2, x2_0, nll = result
        print(result)

        # Upload/update the existing parameter
        params_df.loc[
            (params_df["Patient"] == patient_id) & (params_df["Region"] == "combined"),
            [f"x1_{dataset}", f"x2_{dataset}", f"nll_{dataset}"],
        ] = [x1, x2, nll]

    else:
        ############################### RUN BY REGION ######################################
        if region_num is None:
            print("Error: region_num must be provided when region is True.")
            sys.exit(1)
        else:
            if dataset == "panpep":
                data_dir = f"{root_dir}/data/glioblastoma_data/PANPEP/"
            else:
                data_dir = f"{root_dir}/data/glioblastoma_data/ERGOII/"
            filepaths = os.path.join(data_dir, patient_id, f"{patient_id}_region*.csv")
            files = glob.glob(filepaths)
            sorted_files = sorted(
                files, key=lambda x: int(x.split("region")[1].split(".")[0])
            )
            if region_num == "all":
                # Create a dictionary to store results with their region numbers
                # region_results = {}
                for file in sorted_files:
                    # print(file)
                    match = re.search(
                        r"_region(\d+)\.csv$", file
                    )  # note file is full path
                    if match:
                        region_number = match.group(1)
                    else:
                        print("Region numbers not found.")
                    if dataset == "panpep":
                        patient_data = pd.read_csv(
                            file,
                            sep=",",
                        )
                    elif dataset == "vdjdb":
                        patient_data = pd.read_csv(
                            file,
                            sep=",",
                            usecols=["CDR3", "kr_vdjdb", "counts"],
                        )
                        patient_data.rename(columns={"kr_vdjdb": "kr"}, inplace=True)
                    elif dataset == "mcpas":
                        patient_data = pd.read_csv(
                            file,
                            sep=",",
                            usecols=["CDR3", "kr_mcpas", "counts"],
                        )
                        patient_data.rename(columns={"kr_mcpas": "kr"}, inplace=True)
                    if patient_data.empty:
                        print(f"No data available for file: {file}. Skipping...")
                        continue  # Skip this file if it's empty
                    print(f"Processing Region-{region_number}")
                    clone_count_values = patient_data["counts"].values
                    kr_values = patient_data["kr"].values
                    scaled_kr_values = kr_values / max_kr
                    result = run_optimization(
                        clone_count_values=clone_count_values,
                        scaled_kr_values=scaled_kr_values,
                        fixed_x1=fixed_x1,
                        verbose=verbose,
                    )
                    x1, x2, x2_0, nll = result
                    # region_results[region_number] = result
                    print(result)

                    # Upload/update the existing parameter
                    params_df.loc[
                        (params_df["Patient"] == patient_id)
                        & (params_df["Region"] == f"region{region_number}"),
                        [f"x1_{dataset}", f"x2_{dataset}", f"nll_{dataset}"],
                    ] = [x1, x2, nll]
            else:
                try:
                    region_number = int(region_num)
                    print(f"Processing Region-{region_number}")
                    if dataset == "panpep":
                        data_dir = f"{root_dir}/data/glioblastoma_data/PANPEP/"
                        patient_data = pd.read_csv(
                            f"{data_dir}/{patient_id}/{patient_id}_region{region_number}.csv",
                            sep=",",
                        )
                    elif dataset == "vdjdb":
                        data_dir = f"{root_dir}/data/glioblastoma_data/ERGOII/"
                        patient_data = pd.read_csv(
                            f"{data_dir}/{patient_id}/{patient_id}_region{region_number}.csv",
                            sep=",",
                            usecols=["CDR3", "kr_vdjdb", "counts"],
                        )
                        patient_data.rename(columns={"kr_vdjdb": "kr"}, inplace=True)
                    elif dataset == "mcpas":
                        patient_data = pd.read_csv(
                            f"{data_dir}/{patient_id}/{patient_id}_region{region_number}.csv",
                            sep=",",
                            usecols=["CDR3", "kr_mcpas", "counts"],
                        )
                        patient_data.rename(columns={"kr_mcpas": "kr"}, inplace=True)
                    clone_count_values = patient_data["counts"].values
                    kr_values = patient_data[f"kr"].values
                    scaled_kr_values = kr_values / max_kr
                    result = run_optimization(
                        clone_count_values=clone_count_values,
                        scaled_kr_values=scaled_kr_values,
                        fixed_x1=fixed_x1,
                        verbose=verbose,
                    )
                    x1, x2, x2_0, nll = result
                    print(result)

                    # Upload/update the existing parameter
                    params_df.loc[
                        (params_df["Patient"] == patient_id)
                        & (params_df["Region"] == f"region{region_number}"),
                        [f"x1_{dataset}", f"x2_{dataset}", f"nll_{dataset}"],
                    ] = [x1, x2, nll]
                except:
                    print(
                        "Error: region_num must be an integer or 'all' or the provided region number is not available."
                    )
                    sys.exit(1)

    ############################ Save/UPDATE RESULTS ##################################
    with pd.ExcelWriter(
        results_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        params_df.to_excel(writer, sheet_name="parameters", index=False)

# if __name__ == "__main__":
#     # Add argument parsing for patient_id
#     parser = argparse.ArgumentParser(description="Process patient ID.")
#     parser.add_argument(
#         "--patient_id", type=str, help="The ID of the patient to process"
#     )
#     parser.add_argument(
#         "--dataset", type=str, help="Choose one from panpep or vdjdb or mcpas"
#     )
#     parser.add_argument(
#         "--region",
#         type=str,
#         choices=["True", "False"],
#         default="False",
#         help="Set to True if you want to minimize by regions",
#     )
#     parser.add_argument(
#         "--region_num",
#         type=str,
#         help='If region is true provide region number. If you want to do for all regions at once provide input "all".',
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output during optimization",
#     )  # Added verbose argument
#     args = parser.parse_args()

#     patient_id = args.patient_id
#     dataset = args.dataset
#     region = args.region
#     region_num = args.region_num
#     verbose = args.verbose
#     print(f"Processing data for patient ID: {patient_id}-{dataset}")

#     if dataset == "panpep":
#         full_data = pd.read_csv(
#             f"{root_dir}/data/BrMET_and_GBM_data-PANPEP.csv", sep=","
#         )
#     elif dataset == "vdjdb":
#         full_data = pd.read_csv(
#             f"{root_dir}/data/BrMET_and_GBM_data-ERGO-II.csv",
#             sep=",",
#             usecols=["Patient", "CDR3", "counts", "kr_vdjdb"],
#         )
#         full_data = full_data.rename(columns={"kr_vdjdb": "kr"}, inplace=True)
#     elif dataset == "mcpas":
#         full_data = pd.read_csv(
#             f"{root_dir}/data/BrMET_and_GBM_data-ERGO-II.csv",
#             sep=",",
#             usecols=["Patient", "CDR3", "counts", "kr_mcpas"],
#         )
#         full_data = full_data.rename(columns={"kr_mcpas": "kr"}, inplace=True)
#     else:
#         raise FileNotFoundError(f"Data not found.")

#     max_kr = max(full_data["kr"].values)

#     # Load the results file to upload results
#     results_file = f"{root_dir}/results/results.xlsx"
#     params_df = pd.read_excel(results_file, sheet_name="parameters", engine="openpyxl")

#     if region == "False":
#         patient_data = full_data[full_data["Patient"] == patient_id]
#         if patient_data.empty:
#             print(f"No data available for patient ID: {patient_id}. Skipping...")
#             sys.exit(1)
#         clone_count_values = patient_data["counts"].values
#         kr_values = patient_data["kr"].values
#         scaled_kr_values = kr_values / max_kr
#         result = run_optimization(
#             clone_count_values=clone_count_values,
#             scaled_kr_values=scaled_kr_values,
#             verbose=verbose,
#         )
#         x1, x2, x1_0, x2_0, nll = result
#         print(result)

#         # Upload/update the existing parameter
#         relevant_row = params_df[
#             (params_df["Patient"] == patient_id) & (params_df["Region"] == "combined")
#         ]
#         relevant_row[f"x1_{dataset}"] = x1
#         relevant_row[f"x2_{dataset}"] = x2
#         relevant_row[f"nll_{dataset}"] = nll

#     else:
#         ############################### RUN BY REGION ######################################
#         if region_num is None:
#             print("Error: region_num must be provided when region is True.")
#             sys.exit(1)
#         else:
#             if dataset == "panpep":
#                 data_dir = f"{root_dir}/data/glioblastoma_data/PANPEP/"
#             else:
#                 data_dir = f"{root_dir}/data/glioblastoma_data/ERGOII/"
#             filepaths = os.path.join(data_dir, patient_id, f"{patient_id}_region*.csv")
#             files = glob.glob(filepaths)
#             sorted_files = sorted(
#                 files, key=lambda x: int(x.split("region")[1].split(".")[0])
#             )
#             if region_num == "all":
#                 # Create a dictionary to store results with their region numbers
#                 # region_results = {}
#                 for file in sorted_files:
#                     region_number = file.split("region")[1].split("_")[0][:-4]
#                     if dataset == "panpep":
#                         patient_data = pd.read_csv(
#                             file,
#                             sep=",",
#                         )
#                     elif dataset == "vdjdb":
#                         patient_data = pd.read_csv(
#                             file,
#                             sep=",",
#                             usecols=["CDR3", "kr_vdjdb", "counts"],
#                         )
#                         patient_data = patient_data.rename(columns={"kr_vdjdb": "kr"})
#                     elif dataset == "mcpas":
#                         patient_data = pd.read_csv(
#                             file,
#                             sep=",",
#                             usecols=["CDR3", "kr_mcpas", "counts"],
#                         )
#                         patient_data = patient_data.rename(columns={"kr_mcpas": "kr"})
#                     if patient_data.empty:
#                         print(f"No data available for file: {file}. Skipping...")
#                         continue  # Skip this file if it's empty
#                     print(f"Processing Region-{region_number}")
#                     clone_count_values = patient_data["counts"].values
#                     kr_values = patient_data["kr"].values
#                     scaled_kr_values = kr_values / max_kr
#                     result = run_optimization(
#                         clone_count_values=clone_count_values,
#                         scaled_kr_values=scaled_kr_values,
#                         verbose=verbose,
#                     )
#                     x1, x2, x1_0, x2_0, nll = result
#                     # region_results[region_number] = result
#                     print(result)
#                     # Upload/update the existing parameter
#                     relevant_row = params_df[
#                         (params_df["Patient"] == patient_id)
#                         & (params_df["Region"] == f"region{region_num}")
#                     ]
#                     relevant_row[f"x1_{dataset}"] = x1
#                     relevant_row[f"x2_{dataset}"] = x2
#                     relevant_row[f"nll_{dataset}"] = nll
#             else:
#                 try:
#                     region_number = int(region_num)
#                     print(f"Processing Region-{region_number}")
#                     if dataset == "panpep":
#                         data_dir = f"{root_dir}/data/glioblastoma_data/PANPEP/"
#                         patient_data = pd.read_csv(
#                             f"{data_dir}/{patient_id}/{patient_id}_region{region_num}.csv",
#                             sep=",",
#                         )
#                     elif dataset == "vdjdb":
#                         data_dir = f"{root_dir}/data/glioblastoma_data/ERGOII/"
#                         patient_data = pd.read_csv(
#                             f"{data_dir}/{patient_id}/{patient_id}_region{region_num}.csv",
#                             sep=",",
#                             usecols=["CDR3", "kr_vdjdb", "counts"],
#                         )
#                         patient_data = patient_data.rename(columns={"kr_vdjdb": "kr"})
#                     elif dataset == "mcpas":
#                         patient_data = pd.read_csv(
#                             f"{data_dir}/{patient_id}/{patient_id}_region{region_num}.csv",
#                             sep=",",
#                             usecols=["CDR3", "kr_mcpas", "counts"],
#                         )
#                         patient_data = patient_data.rename(columns={"kr_mcpas": "kr"})
#                     clone_count_values = patient_data["counts"].values
#                     kr_values = patient_data[f"kr"].values
#                     scaled_kr_values = kr_values / max_kr
#                     result = run_optimization(
#                         clone_count_values=clone_count_values,
#                         scaled_kr_values=scaled_kr_values,
#                         verbose=verbose,
#                     )
#                     x1, x2, x1_0, x2_0, nll = result
#                     print(result)
#                     # Upload/update the existing parameter
#                     relevant_row = params_df[
#                         (params_df["Patient"] == patient_id)
#                         & (params_df["Region"] == f"region{region_num}")
#                     ]
#                     relevant_row[f"x1_{dataset}"] = x1
#                     relevant_row[f"x2_{dataset}"] = x2
#                     relevant_row[f"nll_{dataset}"] = nll
#                 except:
#                     print(
#                         "Error: region_num must be an integer or 'all' or the provided region number is not available."
#                     )
#                     sys.exit(1)

#     ############################ Save/UPDATE RESULTS ##################################
#     with pd.ExcelWriter(
#         results_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
#     ) as writer:
#         params_df.to_excel(writer, sheet_name="parameters", index=False)
