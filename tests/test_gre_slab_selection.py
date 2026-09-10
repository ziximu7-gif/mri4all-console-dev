import math

import numpy as np
import pypulseq as pp


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