import math
from pathlib import Path

import numpy as np
import pypulseq as pp

from sequences.common.get_trajectory import choose_pe_order
from common.constants import *
import common.logger as logger
from common.geometry import (
    cm_to_m,
    orientation_channels,
)
from pypulseq.add_gradients import (
    add_gradients,
)
from pypulseq.scale_grad import (
    scale_grad,
)

log = logger.get_logger()
_GRADIENT_AXES = (
    "x",
    "y",
    "z",
)

_GRADIENT_AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def _transform_gradient_events(
    gradients,
    logical_to_scanner,
    system,
):
    """
    Transform logical X/Y/Z gradient events
    into physical scanner X/Y/Z gradients.

    G_scanner = M @ G_logical
    """

    contributions = {
        "x": [],
        "y": [],
        "z": [],
    }

    for gradient in gradients:
        if gradient is None:
            continue

        if gradient.type not in (
            "trap",
            "grad",
        ):
            raise ValueError(
                "Only gradient events can "
                "be transformed."
            )

        logical_axis_index = (
            _GRADIENT_AXIS_INDEX[
                gradient.channel
            ]
        )

        for (
            scanner_axis_index,
            scanner_axis,
        ) in enumerate(
            _GRADIENT_AXES
        ):
            coefficient = float(
                logical_to_scanner[
                    scanner_axis_index,
                    logical_axis_index,
                ]
            )

            if abs(coefficient) < 1e-12:
                continue

            transformed = scale_grad(
                grad=gradient,
                scale=coefficient,
            )

            transformed.channel = (
                scanner_axis
            )

            contributions[
                scanner_axis
            ].append(
                transformed
            )

    scanner_gradients = []

    for scanner_axis in (
        _GRADIENT_AXES
    ):
        axis_gradients = (
            contributions[
                scanner_axis
            ]
        )

        if not axis_gradients:
            continue

        scanner_gradients.append(
            add_gradients(
                grads=axis_gradients,
                system=system,
            )
        )

    return scanner_gradients
def _add_gradient_block(
    seq,
    gradients,
    logical_to_scanner,
    system,
    extra_events=None,
):
    if logical_to_scanner is None:
        output_gradients = list(
            gradients
        )
    else:
        output_gradients = (
            _transform_gradient_events(
                gradients=gradients,
                logical_to_scanner=(
                    logical_to_scanner
                ),
                system=system,
            )
        )

    events = list(
        output_gradients
    )

    if extra_events is not None:
        events.extend(
            extra_events
        )

    seq.add_block(
        *events
    )
def pypulseq_gre3D(
    inputs,
    check_timing,
    output_file,
    working_folder,
):
    """
    Generate a Cartesian 3D GRE Pulseq sequence.

    This is the shared GRE generator used by imaging sequences
    and, later, by B0 mapping adjustments.
    """

    alpha1 = inputs["FA"]
    alpha1_duration = 80e-6

    TR = inputs["TR"] / 1000

    echo_times = inputs.get(
        "echo_times",
        [inputs["TE"]],
    )

    if len(echo_times) not in (1, 2):
        log.error(
            "GRE currently supports one or two echoes."
        )
        return False

    echo_times = [
        float(te) / 1000
        for te in echo_times
    ]

    TE1 = echo_times[0]
    TE2 = (
        echo_times[1]
        if len(echo_times) == 2
        else None
    )

    planned_fov_m = inputs.get(
        "planned_fov_m"
    )
    logical_to_scanner = (
        inputs.get(
            "logical_to_scanner"
        )
    )

    if (
        (planned_fov_m is None)
        !=
        (logical_to_scanner is None)
    ):
        log.error(
            "planned_fov_m and "
            "logical_to_scanner must "
            "be provided together."
        )
        return False

    if planned_fov_m is not None:
        planned_fov_m = np.asarray(
            planned_fov_m,
            dtype=float,
        )

        if planned_fov_m.shape != (3,):
            log.error(
                "planned_fov_m must contain "
                "three FOV dimensions."
            )
            return False

        if np.any(
            planned_fov_m <= 0
        ):
            log.error(
                "Planned FOV dimensions "
                "must be positive."
            )
            return False

        fovx = float(
            planned_fov_m[0]
        )
        fovy = float(
            planned_fov_m[1]
        )
        fovz = float(
            planned_fov_m[2]
        )

        log.info(
            "Using planned GRE FOV [m]: "
            f"{fovx}, {fovy}, {fovz}"
        )

    else:
        base_fov_m = cm_to_m(
            inputs["FOV"]
        )

        fovx = base_fov_m
        fovy = base_fov_m

        # Preserve legacy GRE Z behavior.
        fovz = base_fov_m / 2.0
        
    if logical_to_scanner is not None:
        logical_to_scanner = (
            np.asarray(
                logical_to_scanner,
                dtype=float,
            )
        )

        if (
            logical_to_scanner.shape
            != (3, 3)
        ):
            log.error(
                "logical_to_scanner "
                "must be 3x3."
            )
            return False

        if not np.allclose(
            logical_to_scanner.T
            @ logical_to_scanner,
            np.eye(3),
            atol=1e-6,
        ):
            log.error(
                "logical_to_scanner "
                "must be orthogonal."
            )
            return False


    Nx = inputs["baseresolution"]
    Ny = inputs["baseresolution"]
    Nz = inputs["slices"]

    dim0 = Ny
    dim1 = Nz

    num_averages = inputs["NSA"]
    orientation = inputs["orientation"]
    BW = inputs["BW"]
    ordering = inputs["ordering"]
    dummyshots = inputs["dummy_shots"]

    pe_order_file = (
        Path(working_folder)
        / mri4all_taskdata.RAWDATA
        / mri4all_scanfiles.PE_ORDER
    )

    adc_phase_file = (
        Path(working_folder)
        / mri4all_taskdata.RAWDATA
        / mri4all_scanfiles.ADC_PHASE
    )

    adc_dwell = 1 / BW
    adc_duration = Nx * adc_dwell

    if logical_to_scanner is not None:
        # Build the sequence first in logical
        # read/phase/third coordinates.
        ch0 = "x"
        ch1 = "y"
        ch2 = "z"
    else:
        # Legacy non-planned GRE path.
        ch0, ch1, ch2 = (
            orientation_channels(
                orientation
            )
        )

    seq = pp.Sequence()
    n_shots = int(Ny * Nz)

    system = pp.Opts(
        max_grad=100,
        grad_unit="mT/m",
        max_slew=4000,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        rf_raster_time=1e-6,
        adc_dead_time=20e-6,
    )

    rf1 = pp.make_block_pulse(
        flip_angle=alpha1 * math.pi / 180,
        duration=alpha1_duration,
        delay=0e-6,
        system=system,
        use="excitation",
    )

    delta_kx = 1 / fovx
    delta_ky = 1 / fovy
    delta_kz = 1 / fovz

    gx = pp.make_trapezoid(
        channel=ch0,
        flat_area=Nx * delta_kx,
        flat_time=adc_duration,
        system=system,
    )

    adc = pp.make_adc(
        num_samples=2 * Nx,
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )

    gx_pre = pp.make_trapezoid(
        channel=ch0,
        area=gx.area / 2.0,
        system=system,
    )
    gx_pre.amplitude = -gx_pre.amplitude

    gx_rewind = pp.make_trapezoid(
        channel=ch0,
        area=-gx.area,
        system=system,
    )

    pe_order = choose_pe_order(
        ndims=3,
        npe=[dim0, dim1],
        traj=ordering,
        save_pe_order=True,
        save_path=str(pe_order_file),
    )

    phase_areas0 = (
        pe_order[:, 0]
        * delta_ky
    )

    phase_areas1 = (
        pe_order[:, 1]
        * delta_kz
    )

    gy_pre = pp.make_trapezoid(
        channel=ch1,
        area=1.0 * np.max(phase_areas0),
        system=system,
    )

    gz_pre = pp.make_trapezoid(
        channel=ch2,
        area=-1.0 * np.max(phase_areas1),
        system=system,
    )

    pre_duration = max(
        pp.calc_duration(gy_pre),
        pp.calc_duration(gz_pre),
    )

    pre_duration = max(
        pre_duration,
        pp.calc_duration(gx_pre),
    )

    gx_spoil = pp.make_trapezoid(
        channel=ch0,
        area=Nx * delta_kx,
        system=system,
    )

    if TE1 == 0:
        tau1 = (
            10
            * seq.grad_raster_time
        )

        TE1 = (
            tau1
            + 0.5 * pp.calc_duration(rf1)
            + pre_duration
            + 0.5 * pp.calc_duration(gx)
        )
    else:
        tau1 = (
            math.ceil(
                (
                    TE1
                    - 0.5 * pp.calc_duration(rf1)
                    - pre_duration
                    - 0.5 * pp.calc_duration(gx)
                )
                / seq.grad_raster_time
            )
            * seq.grad_raster_time
        )

    tau2 = None

    if TE2 is not None:
        if TE2 <= TE1:
            log.error(
                "TE2 must be greater than TE1."
            )
            return False

        tau2 = (
            math.ceil(
                (
                    TE2
                    - TE1
                    - pp.calc_duration(gx)
                    - pp.calc_duration(gx_rewind)
                )
                / seq.grad_raster_time
            )
            * seq.grad_raster_time
        )

        if tau2 < 0:
            log.error(
                "TE2 is too short for the second GRE echo."
            )
            return False
        
    last_TE = (
        TE2
        if TE2 is not None
        else TE1
    )

    delay_TR = (
        math.ceil(
            (
                TR
                - 0.5 * pp.calc_duration(rf1)
                - last_TE
                - 0.5 * pp.calc_duration(gx)
                - pp.calc_duration(gx_spoil)
            )
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )

    assert np.all(tau1 >= 0)

    if tau2 is not None:
        assert np.all(tau2 >= 0)

    assert np.all(delay_TR >= 0)
    adc_phase = []

    rfspoil_phase = 0
    rfspoil_inc = 0
    rfspoil_incinc = 117.0

    for avg in range(num_averages):
        for i in range(
            n_shots + dummyshots
        ):
            rfspoil_inc = (
                rfspoil_inc
                + rfspoil_incinc
            )

            rfspoil_phase = (
                rfspoil_phase
                + rfspoil_inc
            )

            rfspoil_phase = np.mod(
                rfspoil_phase,
                360.0,
            )

            rfspoil_inc = np.mod(
                rfspoil_inc,
                360.0,
            )

            is_dummyshot = (
                i < dummyshots
            )

            rf1.phase_offset = (
                rfspoil_phase
                / 180
                * math.pi
            )

            seq.add_block(rf1)

            if is_dummyshot:
                pe_idx = 0
            else:
                pe_idx = (
                    i
                    - dummyshots
                )

            gy_pre = pp.make_trapezoid(
                channel=ch1,
                area=-1.0
                * phase_areas0[pe_idx],
                duration=pre_duration,
                system=system,
            )

            gz_pre = pp.make_trapezoid(
                channel=ch2,
                area=-1.0
                * phase_areas1[pe_idx],
                duration=pre_duration,
                system=system,
            )

            _add_gradient_block(
                seq=seq,
                gradients=[
                    gx_pre,
                    gy_pre,
                    gz_pre,
                ],
                logical_to_scanner=(
                    logical_to_scanner
                ),
                system=system,
            )

            seq.add_block(
                pp.make_delay(tau1)
            )

            if is_dummyshot:
                _add_gradient_block(
                    seq=seq,
                    gradients=[
                        gx,
                    ],
                    logical_to_scanner=(
                        logical_to_scanner
                    ),
                    system=system,
                )
            else:
                _add_gradient_block(
                    seq=seq,
                    gradients=[
                        gx,
                    ],
                    logical_to_scanner=(
                        logical_to_scanner
                    ),
                    system=system,
                    extra_events=[
                        adc,
                    ],
                )

                adc_phase.append(
                    rfspoil_phase
                )
            if TE2 is not None:
                # Rewind readout k-space after echo 1.
                _add_gradient_block(
                    seq=seq,
                    gradients=[
                        gx_rewind,
                    ],
                    logical_to_scanner=(
                        logical_to_scanner
                    ),
                    system=system,
                )

                if tau2 > 0:
                    seq.add_block(
                        pp.make_delay(tau2)
                    )

                # Acquire echo 2 with the same
                # forward readout gradient.
                if is_dummyshot:
                    _add_gradient_block(
                        seq=seq,
                        gradients=[
                            gx,
                        ],
                        logical_to_scanner=(
                            logical_to_scanner
                        ),
                        system=system,
                    )

                else:
                    _add_gradient_block(
                        seq=seq,
                        gradients=[
                            gx,
                        ],
                        logical_to_scanner=(
                            logical_to_scanner
                        ),
                        system=system,
                        extra_events=[
                            adc,
                        ],
                    )

                    adc_phase.append(
                        rfspoil_phase
                    )

            gy_pre.amplitude = (
                -gy_pre.amplitude
            )

            gz_pre.amplitude = (
                -gz_pre.amplitude
            )

            _add_gradient_block(
                seq=seq,
                gradients=[
                    gx_spoil,
                    gy_pre,
                    gz_pre,
                ],
                logical_to_scanner=(
                    logical_to_scanner
                ),
                system=system,
            )

            seq.add_block(
                pp.make_delay(delay_TR)
            )

    if check_timing:
        ok, error_report = (
            seq.check_timing()
        )

        if ok:
            log.info(
                "Timing check passed successfully"
            )
        else:
            log.info(
                "Timing check failed. "
                "Error listing follows:"
            )

            for error in error_report:
                print(error)

    try:
        np.save(
            adc_phase_file,
            adc_phase,
        )
    except Exception:
        log.error(
            "Could not write file with ADC phase"
        )
        return False

    try:
        seq.write(output_file)
        log.debug(
            "Seq file stored"
        )
    except Exception:
        log.error(
            "Could not write sequence file"
        )
        return False

    return True