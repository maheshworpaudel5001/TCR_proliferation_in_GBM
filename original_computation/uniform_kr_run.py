import sys

root_dir = "/home/gddaslab/mxp140/tcr_project_ultimate"
sys.path.append(root_dir)

import warnings

warnings.filterwarnings("ignore")

import itertools
import argparse
import numpy as np
import pandas as pd
from probability import probability
from scipy.optimize import minimize
from mpi4py import MPI  # type: ignore

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


def calc_probs(kr, x1, x2, maxM):
    probabilities = [
        probability(x1, x2, kr, M) for M in range(1, maxM + 1)  # type: ignore
    ]
    return probabilities


def calc_probs_parallel(args):
    kr, x1, x2, maxM = args
    result = calc_probs(kr, x1, x2, maxM)
    return result


def generate_configuration_per_tcr(prob_array, size=1):
    config_per_tcr = np.random.choice(
        range(1, len(prob_array) + 1),
        size=size,
        replace=True,
        p=prob_array / sum(prob_array),
    )
    return config_per_tcr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate probabilities.")
    parser.add_argument("--patient_id", type=str, required=True, help="patient id")
    args = parser.parse_args()
    patient_id = args.patient_id

    full_data = pd.read_csv(f"{root_dir}/data/BrMET_and_GBM_data.csv", sep=",")
    max_kr = np.max(full_data["kr"].values)
    min_kr = np.min(full_data["kr"].values)

    # Load the uniform kr saved as compressed .npz file
    all_uniform_kr_data = np.load(f"{root_dir}/results/uniform_dists.npz")
    uniform_scaled_kr_values = all_uniform_kr_data[patient_id]
    all_uniform_kr_data.close()  # close the file to free memory

    patient_data = full_data[full_data["Patient"] == patient_id]
    clone_count_values = patient_data["counts"]
    n = len(clone_count_values)

    x1 = 100
    bounds = ((1e-10, 100),)
    verbose = False

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

    uniform_scaled_kr_values_splitted = np.array_split(uniform_scaled_kr_values, size)
    uniform_scaled_kr_values_scattered = comm.scatter(
        uniform_scaled_kr_values_splitted, root=0
    )

    results = []
    for uniform_scaled_kr_array in uniform_scaled_kr_values_scattered:
        initial_value = [np.random.uniform(bounds[0][0], bounds[0][1], size=1)[0]]
        local_args = (
            clone_count_values,
            uniform_scaled_kr_array,
            x1,
            bounds,
            initial_value,
            verbose,
        )
        local_result = run_optimization_with_single_argument(local_args)
        results.append(local_result)
    results = comm.gather(results, root=0)
    # save result in your local cpu
    if rank == 0:
        final_result = list(itertools.chain.from_iterable(results))
        output_file = f"{root_dir}/results/{patient_id}_uniform_kr_test_params.csv"
        print(f"Collecting and saving data in file:\n {output_file}")
        sys.stdout.flush()  # Flush the output buffer
        output_df = pd.DataFrame(final_result, columns=["x1", "x2", "x2_0", "nll"])
        output_df.to_csv(output_file, sep=",", index=False)
    MPI.Finalize()
