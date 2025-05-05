import os
import sys
import h5py
import shutil
import argparse
import itertools
import numpy as np
import pandas as pd
from mpi4py import MPI  # type: ignore

root_dir = "/home/gddaslab/mxp140/tcr_project_ultimate"
sys.path.append(root_dir)
from constants import max_kr_mcpas
from pdf import generate_configuration_per_tcr
from calculate_probabilities import calc_probs_for_single_tcr


def calc_probs_parallel(args):
    kr, x1, x2, maxM = args
    result = calc_probs_for_single_tcr(kr, x1, x2, maxM)
    return result


def chunk_datafile(df, number_of_chunks):
    if number_of_chunks <= 0:
        raise ValueError("Number of chunks must be greater than 0.")

    chunk_size = len(df) // number_of_chunks
    indices = [
        range(i * chunk_size, (i + 1) * chunk_size) for i in range(number_of_chunks)
    ]

    # Handle the last chunk if there are remaining rows
    if len(df) % number_of_chunks != 0:
        indices[-1] = range((number_of_chunks - 1) * chunk_size, len(df))

    chunked_dfs = [df.iloc[chunk] for chunk in indices if chunk.stop <= len(df)]
    return chunked_dfs


def save_results(result, filename):
    result_as_array = np.array(result, dtype="float32")
    # Ensure the filename has the .h5 extension
    if not filename.endswith(".h5"):
        filename += ".h5"
    with h5py.File(filename, "w") as file:
        file.create_dataset("result", data=result_as_array)


def combine_h5_files(input_files, output_file, result_key="result"):
    combined_results = []

    for file in input_files:
        with h5py.File(file, "r") as h5_file:
            if result_key in h5_file:
                data = h5_file[result_key][:]  # type: ignore
                combined_results.append(data)
            else:
                print(f"Key '{result_key}' not found in {file}")

    if combined_results:
        combined_results = np.concatenate(combined_results, axis=0)
        if not output_file.endswith(".h5"):
            output_file += ".h5"
        with h5py.File(output_file, "w") as h5_file:
            h5_file.create_dataset(result_key, data=combined_results)
        print(f"Combined results saved to {output_file}")
    else:
        print("No results to combine.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate probabilities.")
    parser.add_argument("--patient_id", type=str, required=True, help="patient id")
    parser.add_argument("--region", type=int, required=True, help="Region number")

    args = parser.parse_args()
    patient_id = args.patient_id
    region = args.region

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

    ##################Prepare data###################
    full_data = pd.read_csv(
        f"{root_dir}/data/BrMET_and_GBM_data-ERGOII.csv",
        sep=",",
    )
    full_data = full_data.rename(columns={"kr_mcpas": "kr"})
    max_kr = max_kr_mcpas
    region = int(region)
    if rank == 0:
        print(f"Processing {patient_id}-region-{region}")
        sys.stdout.flush()  # Flush the output buffer
    data_dir = f"{root_dir}/data/glioblastoma_data/ERGOII"
    filepath = os.path.join(data_dir, patient_id, f"{patient_id}_region{region}.csv")
    patient_data = pd.read_csv(filepath, sep=",")
    patient_data = patient_data.rename(columns={"kr_mcpas": "kr"})
    number_of_chunks = 5
    if patient_data.shape[0] > 1000:
        chunked_patient_data = chunk_datafile(patient_data, number_of_chunks)
    else:
        chunked_patient_data = chunk_datafile(patient_data, 1)

    params_df = pd.read_excel(
        "/home/gddaslab/mxp140/tcr_project_ultimate/results/results.xlsx",
        sheet_name="parameters",
        engine="openpyxl",
    )
    patient_params_df = params_df[params_df["Patient"] == patient_id]
    patient_params_df_by_region = patient_params_df[
        patient_params_df["Region"] == f"region{region}"
    ]
    x1_value = patient_params_df_by_region["x1_mcpas"].values[0]
    x2_value = patient_params_df_by_region["x2_mcpas"].values[0]

    if rank == 0:
        print(f"x1:{x1_value},x2:{x2_value}")
        sys.stdout.flush()  # Flush the output buffer

    M_max_values = {
        "BrMET008": 10000,
        "BrMET009": 10000,
        "BrMET010": 10000,
        "BrMET018": 10000,
        "BrMET019": 10000,
        "BrMET025": 10000,
        "BrMET027": 10000,
        "BrMET028": 10000,
        "GBM032": 10000,
        "GBM052": 10000,
        "GBM055": 10000,
        "GBM056": 10000,
        "GBM059": 10000,
        "GBM062": 10000,
        "GBM063": 10000,
        "GBM064": 10000,
        "GBM070": 10000,
        "GBM074": 10000,
        "GBM079": 10000,
    }
    M_max_value = M_max_values.get(patient_id, 1000)

    # Create a temporary folder to create meta files and later delete it
    if rank == 0:
        try:
            temp_dir = f"{root_dir}/temp_{patient_id}_region{region}"
            os.mkdir(temp_dir)
            print(f"Temporary directory created:\n{temp_dir}")
            sys.stdout.flush()  # Flush the output buffer
        except OSError as error:
            print(error)
            sys.stdout.flush()  # Flush the output buffer

    for i, data_chunk in enumerate(chunked_patient_data):
        if rank == 0:
            print(f"=" * 80)
            print(f"Processing Chunk {i+1}/{len(chunked_patient_data)}")
            print(f"=" * 80)
            sys.stdout.flush()  # Flush the output buffer
        scaled_kr_values = data_chunk["kr"].values / max_kr
        scaled_kr_values_splitted = np.array_split(scaled_kr_values, size)
        scaled_kr_values_scattered = comm.scatter(scaled_kr_values_splitted, root=0)

        results = []
        for scaled_kr_value in scaled_kr_values_scattered:
            local_args = (
                scaled_kr_value,
                x1_value,
                x2_value,
                M_max_value,
            )
            local_result = calc_probs_parallel(local_args)
            results.append(local_result)
        results = comm.gather(results, root=0)
        # save result in your local cpu
        if rank == 0:
            final_result = list(itertools.chain.from_iterable(results))
            result_per_chunk_data_per_kr_chunk = save_results(
                result=final_result,
                filename=f"{temp_dir}/probs_chunk{i+1}",
            )
    if rank == 0:
        output_file = f"{root_dir}/results/{patient_id}_region{region}_model_results_from_mcpas.h5"
        if (
            len(chunked_patient_data) == 1
        ):  # The data was never chunked in this case. So simply relabel the filename.
            os.rename(
                f"{temp_dir}/probs_chunk1.h5",
                output_file,
            )
        else:
            input_files = [
                f"{temp_dir}/probs_chunk{i+1}.h5" for i in range(number_of_chunks)
            ]
            combine_h5_files(
                input_files=input_files,
                output_file=output_file,
            )

        # import the result file saved
        with h5py.File(f"{output_file}", "r+") as f:
            # Rename 'result' key to 'probabilities'
            if "result" in f:
                f.move("result", "probabilities")
            probs = f["probabilities"][:]

            # generate configurations
            size = 1000
            all_configs = generate_configuration_per_tcr(
                probs, size=size, disable_progressbar=True
            )
            original_config = patient_data["counts"].values
            tcr_list = patient_data["CDR3"].values

            # Save the new arrays
            f.create_dataset("model_config", data=all_configs)
            f.create_dataset("data_config", data=original_config)
            f.create_dataset("tcrs", data=tcr_list)

        print(f"Output saved in file:\n {output_file}")
        sys.stdout.flush()  # Flush the output buffer

        # remove the temporary directory
        shutil.rmtree(temp_dir)
    MPI.Finalize()
