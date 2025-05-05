import sys

root_dir = "/home/gddaslab/mxp140/tcr_project_ultimate"
sys.path.append(root_dir)

import warnings

warnings.filterwarnings("ignore")

import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import qmc

# from sklearn.utils import resample
from probability import probability
from scipy.optimize import minimize
from mpi4py import MPI  # type: ignore
from minimization import run_optimization  # type: ignore

probability = np.vectorize(probability)


def new_neg_likelihood(
    params, fixed_x1, clone_count_values, scaled_kr_values, verbose=False
):
    # params is now just x2, fixed_x1 is provided separately
    x2 = params[0]  # params is now a 1D array with just x2
    x1 = fixed_x1
    n = len(clone_count_values)
    x1_values = np.full(n, x1)
    x2_values = np.full(n, x2)
    probs = probability(x1_values, x2_values, scaled_kr_values, clone_count_values)
    # Replace zero values with the smallest positive value allowed in Python
    smallest_positive_value = np.finfo(float).eps
    probabilities = np.where(probs == 0, smallest_positive_value, probs)
    sum_log_probs = np.sum(np.log(probabilities))
    neg_sum = -sum_log_probs

    if verbose:
        print(f"Neg-logL: {neg_sum:.8f}")
        print(f"x2: {x2:.8f}")
        print(f"=" * 80)
    return neg_sum


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


def run_optimization_with_single_argument(args):
    count, kr, x1, bounds, x20, verbose = args
    result = run_optimization(count, kr, x1, bounds, x20, verbose)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate probabilities.")
    parser.add_argument("--patient_id", type=str, required=True, help="patient id")
    # parser.add_argument("--n", type=int, required=True, help="Number of initial values")
    args = parser.parse_args()

    # full_data = pd.read_csv(f"{root_dir}/data/BrMET_and_GBM_data.csv", sep=",")
    # max_kr = max(full_data["kr"].values)
    # patient_id = args.patient_id
    # patient_data = full_data[full_data["Patient"] == patient_id]
    # clone_count_values = patient_data["counts"].values
    # kr_values = patient_data["kr"].values
    # scaled_kr_values = kr_values / max_kr
    # x1 = 100  # fix x1
    # bounds = ((1e-10, 100),)
    # num_initial_values = args.n

    # # Use Latin Hypercube Sampling for better parameter space coverage
    # sampler = qmc.LatinHypercube(d=len(bounds))
    # sample = sampler.random(n=num_initial_values)

    # # Scale the samples to our bounds
    # initial_values = qmc.scale(
    #     sample, np.array(bounds)[:, 0], np.array(bounds)[:, 1]  # lower bounds
    # )  # upper bounds
    # # print(initial_values)
    # # import matplotlib.pyplot as plt
    # # plt.plot(initial_values[:, 0], initial_values[:, 1], 'b.')
    # # plt.show()

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # if not MPI.Is_initialized():
    #     if rank == 0:
    #         print(
    #             "MPI is not initialized. Make sure you have OpenMPI installed and properly configured."
    #         )
    #         sys.stdout.flush()  # Flush the output buffer
    #     MPI.Finalize()

    # if rank == 0:
    #     print("MPI is initialized and running.")
    #     sys.stdout.flush()  # Flush the output buffer

    # initial_values_splitted = np.array_split(initial_values, size)
    # initial_values_scattered = comm.scatter(initial_values_splitted, root=0)

    # results = []
    # for x2 in initial_values_scattered:
    #     local_args = (clone_count_values, scaled_kr_values, x1, bounds, x2, False)
    #     local_result = run_optimization_with_single_argument(local_args)
    #     results.append(local_result)
    # results = comm.gather(results, root=0)
    # # save result in your local cpu
    # if rank == 0:
    #     final_result = list(itertools.chain.from_iterable(results))
    #     output_file = f"{root_dir}/results/{patient_id}_multiple_optimizations.csv"
    #     print(f"Collecting and saving data in file:\n {output_file}")
    #     sys.stdout.flush()  # Flush the output buffer
    #     output_df = pd.DataFrame(final_result, columns=["x1", "x2", "x2_0", "nll"])
    #     output_df.to_csv(output_file, sep=",", index=False)
    # MPI.Finalize()

    # ============================ Shuffled case ============================= #
    full_data = pd.read_csv(f"{root_dir}/data/BrMET_and_GBM_data-PANPEP.csv", sep=",")
    max_kr = max(full_data["kr"].values)
    patient_id = args.patient_id
    patient_data = full_data[full_data["Patient"] == patient_id]
    all_bootstrapped_samples = pd.read_csv(
        f"{root_dir}/results/{patient_id}_bootstrapped_samples.csv.gz",
        compression="gzip",
    )
    # patient_data = pd.read_csv(
    #     f"{root_dir}/results/{patient_id}_shuffled_clone_counts.csv.gz",
    #     compression="gzip",
    # )
    # all_clone_count_values = patient_data.loc[:, "counts_1":"counts_5024"].values
    # kr_values = patient_data["kr"].values
    # scaled_kr_values = kr_values / max_kr
    x1 = 100  # fix x1
    bounds = ((1e-10, 100),)
    # Use Latin Hypercube Sampling for better parameter space coverage
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample = sampler.random(n=all_bootstrapped_samples.shape[1])

    # Scale the samples to our bounds
    initial_values = qmc.scale(
        sample, np.array(bounds)[:, 0], np.array(bounds)[:, 1]  # lower bounds
    )  # upper bounds

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not MPI.Is_initialized():
        if rank == 0:
            print(
                "MPI is not initialized. Make sure you have OpenMPI installed and properly configured."
            )
            sys.stdout.flush()  # Flush the output buffer
        MPI.Finalize()

    if rank == 0:
        print("MPI is initialized and running.")
        sys.stdout.flush()  # Flush the output buffer

    sample_names = all_bootstrapped_samples.columns
    sample_names_splitted = np.array_split(sample_names, size)
    sample_names_scattered = comm.scatter(sample_names_splitted, root=0)
    initial_values_splitted = np.array_split(initial_values, size)
    initial_values_scattered = comm.scatter(initial_values_splitted, root=0)

    # Print the rank and the samples received by each process
    # print(f"Process {rank} received sample names: {sample_names_scattered}")

    results = []
    for sample, x2_0 in zip(sample_names_scattered, initial_values_scattered):
        bootstrapped_samples = all_bootstrapped_samples[
            [sample]
        ]  # Use double square brackets
        clone_count_values_list = []
        scaled_kr_values_list = []
        for col in bootstrapped_samples.columns:
            chosen_sample = pd.Series(bootstrapped_samples[col]).value_counts()
            chosen_sample_df = pd.DataFrame(
                {"CDR3": chosen_sample.index, "counts": chosen_sample.values}
            )
            # # Merge with the original dataframe to get the kr values
            chosen_sample_df_kr_added = chosen_sample_df.merge(
                patient_data[["CDR3", "kr"]], on="CDR3", how="left"
            )
            clone_counts_values = chosen_sample_df_kr_added["counts"].values
            scaled_kr_values = chosen_sample_df_kr_added["kr"].values / max_kr
            clone_count_values_list.append(clone_counts_values)
            scaled_kr_values_list.append(scaled_kr_values)

        local_args = (
            clone_count_values_list,
            scaled_kr_values_list,
            x1,
            bounds,
            x2_0,
            False,
        )
        local_result = run_optimization_with_single_argument(local_args)
        results.append(local_result)
    results = comm.gather(results, root=0)
    # save result in your local cpu
    if rank == 0:
        final_result = list(itertools.chain.from_iterable(results))
        output_file = f"{root_dir}/results/{patient_id}_multiple_optimizations_from_bootstrapped_samples.csv"
        print(f"Collecting and saving data in file:\n {output_file}")
        sys.stdout.flush()  # Flush the output buffer
        output_df = pd.DataFrame(final_result, columns=["x1", "x2", "x2_0", "nll"])
        output_df.to_csv(output_file, sep=",", index=False)
    MPI.Finalize()
