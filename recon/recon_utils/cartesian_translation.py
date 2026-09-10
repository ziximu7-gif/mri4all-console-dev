import numpy as np


def apply_cartesian_center_translation(
    kspace,
    center_logical_m,
    fov_logical_m,
    oversampling_read=1,
    center_indices=None,
):
    """
    Re-center Cartesian k-space around a planned
    physical FOV center.

    Coordinate order:
        read, phase, third

    center_logical_m:
        Planned center position along logical
        read/phase/third axes [m].

    fov_logical_m:
        Planned FOV along logical
        read/phase/third axes [m].

    The phase sign here follows the current
    reconstruction convention using np.fft.fftn().
    """

    kspace = np.asarray(
        kspace
    )

    center_logical_m = np.asarray(
        center_logical_m,
        dtype=float,
    )

    fov_logical_m = np.asarray(
        fov_logical_m,
        dtype=float,
    )

    if kspace.ndim != 3:
        raise ValueError(
            "Cartesian k-space must be 3D"
        )

    if center_logical_m.shape != (3,):
        raise ValueError(
            "center_logical_m must contain "
            "read/phase/third"
        )

    if fov_logical_m.shape != (3,):
        raise ValueError(
            "fov_logical_m must contain "
            "read/phase/third"
        )

    if not np.all(
        np.isfinite(center_logical_m)
    ):
        raise ValueError(
            "FOV center must be finite"
        )

    if (
        not np.all(
            np.isfinite(fov_logical_m)
        )
        or np.any(
            fov_logical_m <= 0
        )
    ):
        raise ValueError(
            "FOV dimensions must be "
            "finite and positive"
        )

    if oversampling_read <= 0:
        raise ValueError(
            "oversampling_read must be positive"
        )

    if center_indices is None:
        center_indices = np.array(
            [
                size // 2
                for size in kspace.shape
            ],
            dtype=float,
        )
    else:
        center_indices = np.asarray(
            center_indices,
            dtype=float,
        )

    if center_indices.shape != (3,):
        raise ValueError(
            "center_indices must contain "
            "read/phase/third"
        )

    # Readout is oversampled. Therefore its
    # discrete k-space spacing corresponds to
    # an effective image FOV enlarged by the
    # oversampling factor.
    sampled_fov_m = np.array(
        [
            fov_logical_m[0]
            * float(oversampling_read),
            fov_logical_m[1],
            fov_logical_m[2],
        ],
        dtype=float,
    )

    k_read = (
        np.arange(
            kspace.shape[0],
            dtype=float,
        )
        - center_indices[0]
    ) / sampled_fov_m[0]

    k_phase = (
        np.arange(
            kspace.shape[1],
            dtype=float,
        )
        - center_indices[1]
    ) / sampled_fov_m[1]

    k_third = (
        np.arange(
            kspace.shape[2],
            dtype=float,
        )
        - center_indices[2]
    ) / sampled_fov_m[2]

    read_correction = np.exp(
        -2j
        * np.pi
        * k_read
        * center_logical_m[0]
    )[:, None, None]

    phase_correction = np.exp(
        -2j
        * np.pi
        * k_phase
        * center_logical_m[1]
    )[None, :, None]

    third_correction = np.exp(
        -2j
        * np.pi
        * k_third
        * center_logical_m[2]
    )[None, None, :]

    return (
        kspace
        * read_correction
        * phase_correction
        * third_correction
    )