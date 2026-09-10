import math

import numpy as np
import pypulseq as pp
from sequences.common.gradient_transform import (
    transform_gradient_events,
)
from sequences.common.rf_positioning import (
    apply_centered_rf_frequency_modulation,
)



def test_gre_slab_selective_rf():
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

    slab_thickness_m = 0.08

    rf, gslab, gslab_rephase = (
        pp.make_sinc_pulse(
            flip_angle=(
                20
                * math.pi
                / 180
            ),
            duration=2e-3,
            slice_thickness=(
                slab_thickness_m
            ),
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True,
            use="excitation",
        )
    )

    assert rf.type == "rf"
    assert gslab.type == "trap"
    assert gslab.channel == "z"

    expected_bandwidth_hz = (
        4.0 / 2e-3
    )

    expected_gradient_hz_per_m = (
        expected_bandwidth_hz
        / slab_thickness_m
    )

    np.testing.assert_allclose(
        gslab.amplitude,
        expected_gradient_hz_per_m,
        rtol=1e-6,
    )

    assert (
        gslab_rephase.channel
        == "z"
    )
def test_oblique_slab_gradient_rotation():
    system = pp.Opts(
        max_grad=100,
        grad_unit="mT/m",
        max_slew=4000,
        slew_unit="T/m/s",
    )

    gslab = pp.make_trapezoid(
        channel="z",
        amplitude=1000.0,
        flat_time=1e-3,
        system=system,
    )

    logical_to_scanner = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    transformed = (
        transform_gradient_events(
            gradients=[
                gslab,
            ],
            logical_to_scanner=(
                logical_to_scanner
            ),
            system=system,
        )
    )

    assert len(transformed) == 1

    gradient = transformed[0]

    assert gradient.channel == "y"

    np.testing.assert_allclose(
        gradient.amplitude,
        -1000.0,
    )
def test_off_center_slab_rf_modulation():
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

    slab_thickness_m = 0.08
    center_third_m = 0.02

    rf, gslab, _ = (
        pp.make_sinc_pulse(
            flip_angle=(
                20
                * math.pi
                / 180
            ),
            duration=2e-3,
            slice_thickness=(
                slab_thickness_m
            ),
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True,
            use="excitation",
        )
    )

    original_signal = np.array(
        rf.signal,
        dtype=np.complex128,
        copy=True,
    )

    expected_frequency_hz = (
        gslab.amplitude
        * center_third_m
    )

    apply_centered_rf_frequency_modulation(
        rf=rf,
        frequency_hz=(
            expected_frequency_hz
        ),
    )

    active = (
        np.abs(
            original_signal
        )
        > 1e-6
        * np.max(
            np.abs(
                original_signal
            )
        )
    )

    phase_difference = np.unwrap(
        np.angle(
            rf.signal[active]
            / original_signal[active]
        )
    )

    time_s = (
        rf.t[active]
    )

    phase_slope = np.polyfit(
        time_s,
        phase_difference,
        1,
    )[0]

    measured_frequency_hz = (
        phase_slope
        / (
            2
            * np.pi
        )
    )

    np.testing.assert_allclose(
        measured_frequency_hz,
        expected_frequency_hz,
        rtol=1e-4,
    )

    assert rf.freq_offset == 0.0