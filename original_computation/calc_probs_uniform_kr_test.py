from calc_probs_mpi import *

root_dir = "/home/gddaslab/mxp140/tcr_project_ultimate"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate probabilities.")
    parser.add_argument("--patient_id", type=str, required=True, help="patient id")
    args = parser.parse_args()
    patient_id = args.patient_id
    # output_filename = args.o

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
        f"{root_dir}/data/BrMET_and_GBM_data.csv",
        sep=",",
    )
    # max_kr = max(full_data["kr"].values) # The uniform kr are already divided by max_kr.
    all_uniform_kr_data = np.load(f"{root_dir}/results/uniform_dists.npz")
    uniform_scaled_kr_values = all_uniform_kr_data[patient_id]
    all_uniform_kr_data.close()  # close the file to free memory
    patient_data = full_data[full_data["Patient"] == patient_id]
    # replace the original kr values by uniform kr values
    params_df = pd.read_csv(
        f"{root_dir}/results/{patient_id}_uniform_kr_test_params.csv",
        sep=",",
    )
    x1_value = 100
    which_dist_index = params_df[
        params_df["nll"] == params_df["nll"].min()
    ].index.values[0]
    # th row distribution corresponding to the least nll value
    x2_value = params_df.loc[which_dist_index, "x2"]

    patient_data["kr"] = uniform_scaled_kr_values[which_dist_index, :]
    number_of_chunks = 5
    if patient_data.shape[0] > 1000:
        chunked_patient_data = chunk_datafile(patient_data, number_of_chunks)
    else:
        chunked_patient_data = chunk_datafile(patient_data, 1)

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
            temp_dir = f"temp_{patient_id}"
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
        scaled_kr_values = data_chunk[
            "kr"
        ].values  # / max_kr. Note uniform krs were already divided by max_kr when saving.
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
        output_file = f"/home/gddaslab/mxp140/tcr_project_ultimate/results/probabilities/probs_{patient_id}_uni{which_dist_index}_test.h5"
        print(f"Collecting and saving data in file:\n {output_file}")
        sys.stdout.flush()  # Flush the output buffer
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
        shutil.rmtree(temp_dir)
    MPI.Finalize()
