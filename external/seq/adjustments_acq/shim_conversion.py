import numpy as np


def b0_fit_to_shim_delta(
    fit_result,
    calibration_matrix,
):
    """
    Convert fitted first-order B0 gradients [Hz/m]
    to normalized shim corrections.

    calibration_matrix maps normalized shim changes
    to measured B0 gradient changes:

        delta_gradient = K @ delta_shim

    Therefore the correction solves:

        K @ delta_shim = -measured_gradient
    """

    gradients = np.asarray(
        [
            fit_result[
                "gradient_x_hz_per_m"
            ],
            fit_result[
                "gradient_y_hz_per_m"
            ],
            fit_result[
                "gradient_z_hz_per_m"
            ],
        ],
        dtype=float,
    )

    calibration_matrix = np.asarray(
        calibration_matrix,
        dtype=float,
    )

    if calibration_matrix.shape != (3, 3):
        raise ValueError(
            "Shim calibration matrix must be 3x3."
        )

    if not np.all(np.isfinite(gradients)):
        raise ValueError(
            "B0 gradients must be finite."
        )

    if not np.all(
        np.isfinite(calibration_matrix)
    ):
        raise ValueError(
            "Shim calibration matrix must be finite."
        )

    if np.linalg.matrix_rank(
        calibration_matrix
    ) < 3:
        raise ValueError(
            "Shim calibration matrix is singular."
        )

    delta_shim = np.linalg.solve(
        calibration_matrix,
        -gradients,
    )

    return {
        "shim_x": float(delta_shim[0]),
        "shim_y": float(delta_shim[1]),
        "shim_z": float(delta_shim[2]),
    }