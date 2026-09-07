import json
from pathlib import Path

from external.seq.adjustments_acq.shim_conversion import (
    b0_fit_to_shim_delta,
)


def load_shim_calibration():
    """
    Load scanner-specific shim calibration matrix.
    """

    calibration_file = (
        Path(__file__).parents[2]
        / "external"
        / "seq"
        / "adjustments_acq"
        / "shim_calibration.json"
    )

    if not calibration_file.exists():
        raise FileNotFoundError(
            "Shim calibration file not found."
        )

    with open(calibration_file, "r") as f:
        data = json.load(f)

    return data["matrix"]



def calculate_shim_correction(
    b0_fit,
):
    """
    Convert first-order B0 fit result
    into shim correction commands.
    """

    calibration_matrix = (
        load_shim_calibration()
    )

    return b0_fit_to_shim_delta(
        b0_fit,
        calibration_matrix,
    )
def apply_shim_to_task(
    task,
):
    """
    Calculate shim correction and store
    it into ScanTask.
    """

    if "b0_fit" not in task.other:
        raise ValueError(
            "No B0 fit result found."
        )

    correction = calculate_shim_correction(
        task.other["b0_fit"]
    )

    task.other[
        "shim_correction"
    ] = correction

    return correction

def apply_shim_correction(
    task,
):
    """
    Apply shim correction to current shim state.
    """

    correction = task.other.get(
        "shim_correction"
    )

    if correction is None:
        raise ValueError(
            "No shim correction found."
        )

    task.adjustment.shim.shim_x += (
        correction["shim_x"]
    )

    task.adjustment.shim.shim_y += (
        correction["shim_y"]
    )

    task.adjustment.shim.shim_z += (
        correction["shim_z"]
    )

    return task.adjustment.shim
