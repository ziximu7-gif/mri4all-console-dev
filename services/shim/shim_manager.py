import json
import math
from pathlib import Path

import common.logger as logger

from external.seq.adjustments_acq.shim_conversion import (
    b0_fit_to_shim_delta,
)
from sequences.common.util import (
    reading_json_parameter,
    writing_json_parameter,
)


log = logger.get_logger()

SHIM_KEYS = (
    "shim_x",
    "shim_y",
    "shim_z",
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


def calculate_shim_correction(b0_fit):
    """
    Convert first-order B0 fit [Hz/m]
    into normalized shim delta.
    """

    calibration_matrix = load_shim_calibration()

    return b0_fit_to_shim_delta(
        b0_fit,
        calibration_matrix,
    )


def _task_shim_snapshot(task):
    """
    Shim values actually used when this task was acquired.
    """

    return {
        "shim_x": float(
            task.adjustment.shim.shim_x
        ),
        "shim_y": float(
            task.adjustment.shim.shim_y
        ),
        "shim_z": float(
            task.adjustment.shim.shim_z
        ),
    }


def _config_shim(config_data):
    return {
        "shim_x": float(
            config_data.shim_parameters.shim_x
        ),
        "shim_y": float(
            config_data.shim_parameters.shim_y
        ),
        "shim_z": float(
            config_data.shim_parameters.shim_z
        ),
    }


def _validate_absolute_shim(shim_values):
    """
    Basic normalized shim validation.

    Final combined gradient + shim limit is still
    checked by scripts.py::shim().
    """

    for key, value in shim_values.items():

        if not math.isfinite(value):
            raise ValueError(
                f"{key} is not finite: {value}"
            )

        if abs(value) >= 1.0:
            raise ValueError(
                f"{key}={value} exceeds "
                "normalized shim range (-1, 1)."
            )


def process_b0_shim(task):
    """
    Convert B0 fit into a shim correction and persist
    the resulting absolute scanner shim.

    task.adjustment.shim is treated as the acquisition
    snapshot and is NOT modified.
    """

    if "b0_fit" not in task.other:
        raise ValueError(
            "No B0 fit result found."
        )

    correction = calculate_shim_correction(
        task.other["b0_fit"]
    )

    # The shim that was actually active during B0 acquisition.
    before = _task_shim_snapshot(task)

    # Read current persisted scanner state.
    config_data = reading_json_parameter()
    current = _config_shim(config_data)

    # Do not apply a correction derived from stale acquisition
    # conditions if someone changed shim in the meantime.
    for key in SHIM_KEYS:
        if not math.isclose(
            current[key],
            before[key],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Scanner shim changed after B0 acquisition. "
                f"{key}: acquired={before[key]}, "
                f"current={current[key]}. "
                "Refusing to apply stale B0 correction."
            )

    after = {
        key: before[key] + correction[key]
        for key in SHIM_KEYS
    }

    _validate_absolute_shim(after)

    config_data.shim_parameters.shim_x = (
        after["shim_x"]
    )
    config_data.shim_parameters.shim_y = (
        after["shim_y"]
    )
    config_data.shim_parameters.shim_z = (
        after["shim_z"]
    )

    writing_json_parameter(
        config_data=config_data
    )

    result = {
        "before": before,
        "delta": correction,
        "after": after,
    }

    task.other["shim_correction"] = correction
    task.other["shim_result"] = result

    log.info(
        "Applied first-order B0 shim correction: "
        f"before={before}, "
        f"delta={correction}, "
        f"after={after}"
    )

    return result