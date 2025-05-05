import sys
import tqdm
import warnings
import numpy as np
from probability import probability
from concurrent.futures import ProcessPoolExecutor

root_dir = "/home/gddaslab/mxp140/tcr_project_ultimate"
sys.path.append(root_dir)

# Suppress only IntegrationWarning
warnings.filterwarnings("ignore")


def calc_probs_for_single_tcr(kr, x1, x2, maxM):
    return [probability(x1, x2, kr, M) for M in range(1, maxM + 1)]


def calc_probs_for_every_tcr(kr, x1, x2, maxM, disable_progressbar=False):
    if isinstance(kr, np.ndarray) and kr.ndim == 1:
        with ProcessPoolExecutor() as executor:
            # Submit all tasks and get futures
            futures = [
                executor.submit(calc_probs_for_single_tcr, k, x1, x2, maxM) for k in kr
            ]
            # Use tqdm to track completion of futures
            results = list(
                tqdm.tqdm(
                    (future.result() for future in futures),
                    total=len(kr),
                    desc="Processing TCRs",
                    disable=disable_progressbar,
                )
            )
        return np.array(results)
    else:
        raise ValueError("kr must be 1D array or a 1D list.")
