import numpy as np


def reconstruct_cartesian_3d_complex(
    raw,
    order,
    adc_phases,
    dims,
    echo_count=1,
    oversampling_read=2,
):
    """
    Reconstruct one or more complex 3D GRE echoes.

    Parameters
    ----------
    raw:
        Flat complex ADC stream.

    order:
        PE ordering table with one entry per k-space line.

    adc_phases:
        RF-spoiling phase recorded once per ADC event, in degrees.

    dims:
        Tuple/list:
            (num_slices, num_phase, num_readout)

        num_readout includes readout oversampling.

    echo_count:
        Number of echoes acquired per PE line.

    oversampling_read:
        Readout oversampling factor.

    Returns
    -------
    list[np.ndarray]
        One complex 3D image volume per echo.
    """

    num_slices = int(dims[0])
    num_phase = int(dims[1])
    num_readout = int(dims[2])

    num_lines = num_slices * num_phase

    raw = np.asarray(raw)
    order = np.asarray(order)
    adc_phases = np.asarray(adc_phases)

    expected_samples = (
        num_lines
        * echo_count
        * num_readout
    )

    if raw.size != expected_samples:
        raise ValueError(
            "Unexpected GRE raw-data size: "
            f"expected {expected_samples}, "
            f"got {raw.size}."
        )

    if len(order) != num_lines:
        raise ValueError(
            "Unexpected PE-order length: "
            f"expected {num_lines}, "
            f"got {len(order)}."
        )

    expected_adc_phases = (
        num_lines
        * echo_count
    )

    if len(adc_phases) != expected_adc_phases:
        raise ValueError(
            "Unexpected ADC-phase count: "
            f"expected {expected_adc_phases}, "
            f"got {len(adc_phases)}."
        )

    # Acquisition order:
    #
    # line 0 echo 0
    # line 0 echo 1
    # line 1 echo 0
    # line 1 echo 1
    # ...
    raw = raw.reshape(
        num_lines,
        echo_count,
        num_readout,
    )

    adc_phases = adc_phases.reshape(
        num_lines,
        echo_count,
    )

    center_slice = (
        num_slices
        - int(num_slices / 2)
    )

    center_phase = (
        num_phase
        - int(num_phase / 2)
    )

    images = []
    kspaces = []

    for echo_index in range(echo_count):

        kspace = np.zeros(
            (
                num_readout,
                num_phase,
                num_slices,
            ),
            dtype=np.complex128,
        )

        for line_index, line in enumerate(order):

            adc_phase = (
                adc_phases[
                    line_index,
                    echo_index,
                ]
                / 180.0
                * np.pi
            )

            phase_index = (
                center_phase
                - int(line[0])
            ) % num_phase

            slice_index = (
                center_slice
                - int(line[1])
            ) % num_slices

            kspace[
                :,
                phase_index,
                slice_index,
            ] = (
                raw[
                    line_index,
                    echo_index,
                    :,
                ]
                * np.exp(
                    1j * adc_phase
                )
            )
        kspaces.append(kspace)

        image = np.fft.fftshift(
            np.fft.fftn(
                np.fft.fftshift(
                    kspace
                )
            )
        )

        # Keep the same empirical readout phase correction
        # currently used by basic3d reconstruction.
        base_res = image.shape[0]

        for sample in range(base_res):
            image[
                sample,
                :,
                :,
            ] *= np.exp(
                np.pi * 1j
                + (
                    sample
                    - base_res / 2
                )
                / 32
                * np.pi
                * 1j
            )

        if oversampling_read > 0:
            offset = num_readout / 4

            image = image[
                int(offset):int(3 * offset),
                :,
                :,
            ]

        images.append(image)

    return images, kspaces