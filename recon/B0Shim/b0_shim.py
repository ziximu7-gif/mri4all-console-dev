import numpy as np


def calculate_b0_map(
    image_te1: np.ndarray,
    image_te2: np.ndarray,
    delta_te_s: float,
) -> np.ndarray:
    """
    Calculate a B0 off-resonance map from two complex images.

    Parameters
    ----------
    image_te1:
        Complex image acquired at TE1.

    image_te2:
        Complex image acquired at TE2.

    delta_te_s:
        TE2 - TE1, in seconds.

    Returns
    -------
    np.ndarray
        B0 frequency offset map in Hz.
    """

    if image_te1.shape != image_te2.shape:
        raise ValueError(
            "TE1 and TE2 images must have the same shape."
        )

    if delta_te_s <= 0:
        raise ValueError(
            "delta_te_s must be greater than zero."
        )

    phase_difference = np.angle(
        image_te2
        * np.conj(image_te1)
    )

    b0_hz = (
        phase_difference
        / (
            2.0
            * np.pi
            * delta_te_s
        )
    )

    return b0_hz


def fit_first_order_b0(
    b0_hz: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    roi_mask: np.ndarray,
) -> dict:
    """
    Fit a first-order B0 field inside an ROI.

    Model:
        B0(x, y, z)
        =
        offset
        + gx * x
        + gy * y
        + gz * z

    Parameters
    ----------
    b0_hz:
        B0 field map in Hz.

    x_m, y_m, z_m:
        Physical voxel coordinates in meters.
        Must have the same shape as b0_hz.

    roi_mask:
        Boolean mask defining the shim ROI.

    Returns
    -------
    dict
        Fitted offset and first-order gradients.
    """

    if not (
        b0_hz.shape
        == x_m.shape
        == y_m.shape
        == z_m.shape
        == roi_mask.shape
    ):
        raise ValueError(
            "B0 map, coordinates, and ROI mask "
            "must have the same shape."
        )

    mask = np.asarray(
        roi_mask,
        dtype=bool,
    )

    valid = (
        mask
        & np.isfinite(b0_hz)
        & np.isfinite(x_m)
        & np.isfinite(y_m)
        & np.isfinite(z_m)
    )

    voxel_count = int(
        np.count_nonzero(valid)
    )

    if voxel_count < 4:
        raise ValueError(
            "Shim ROI must contain at least "
            "4 valid voxels."
        )

    A = np.column_stack(
        (
            np.ones(voxel_count),
            x_m[valid],
            y_m[valid],
            z_m[valid],
        )
    )

    b = b0_hz[valid]

    coefficients, _, _, _ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    fitted = A @ coefficients
    residual = b - fitted

    return {
        "offset_hz": float(
            coefficients[0]
        ),
        "gradient_x_hz_per_m": float(
            coefficients[1]
        ),
        "gradient_y_hz_per_m": float(
            coefficients[2]
        ),
        "gradient_z_hz_per_m": float(
            coefficients[3]
        ),
        "rms_residual_hz": float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        "voxel_count": voxel_count,
    }