import numpy as np
import pypulseq as pp

from sequences.common.gradient_transform import (
    transform_gradient_events,
)
from common.geometry import (
    planning_euler_to_matrix,
)

from sequences.common.gradient_transform import (
    logical_gradient_safety_scale,
    transform_gradient_events,
)
from pypulseq.add_gradients import (
    add_gradients,
)

def _make_system():
    return pp.Opts(
        max_grad=100,
        grad_unit="mT/m",
        max_slew=4000,
        slew_unit="T/m/s",
    )


def test_gradient_transform_identity():
    system = _make_system()

    gx = pp.make_trapezoid(
        channel="x",
        amplitude=1000.0,
        flat_time=1e-3,
        rise_time=100e-6,
       
        system=system,
    )

    transformed = (
        transform_gradient_events(
            gradients=[gx],
            logical_to_scanner=np.eye(3),
            system=system,
        )
    )

    assert len(transformed) == 1
    assert transformed[0].channel == "x"

    np.testing.assert_allclose(
        transformed[0].amplitude,
        gx.amplitude,
    )


def test_gradient_transform_z_rotation():
    system = _make_system()

    gx = pp.make_trapezoid(
        channel="x",
        amplitude=1000.0,
        flat_time=1e-3,
        rise_time=100e-6,
        
        system=system,
    )

    angle = np.deg2rad(30.0)

    rotation = np.array(
        [
            [
                np.cos(angle),
                -np.sin(angle),
                0.0,
            ],
            [
                np.sin(angle),
                np.cos(angle),
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )

    transformed = (
        transform_gradient_events(
            gradients=[gx],
            logical_to_scanner=rotation,
            system=system,
        )
    )

    by_channel = {
        gradient.channel: gradient
        for gradient in transformed
    }

    assert set(by_channel) == {
        "x",
        "y",
    }

    np.testing.assert_allclose(
        by_channel["x"].amplitude,
        gx.amplitude
        * np.cos(angle),
    )

    np.testing.assert_allclose(
        by_channel["y"].amplitude,
        gx.amplitude
        * np.sin(angle),
    )


def test_gradient_transform_combines_axes():
    system = _make_system()

    gx = pp.make_trapezoid(
        channel="x",
        amplitude=1000.0,
        flat_time=1e-3,
        rise_time=100e-6,
        
        system=system,
    )

    gy = pp.make_trapezoid(
        channel="y",
        amplitude=400.0,
        flat_time=1e-3,
        rise_time=100e-6,
        
        system=system,
    )

    angle = np.deg2rad(30.0)

    rotation = np.array(
        [
            [
                np.cos(angle),
                -np.sin(angle),
                0.0,
            ],
            [
                np.sin(angle),
                np.cos(angle),
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )

    transformed = (
        transform_gradient_events(
            gradients=[
                gx,
                gy,
            ],
            logical_to_scanner=rotation,
            system=system,
        )
    )

    by_channel = {
        gradient.channel: gradient
        for gradient in transformed
    }

    expected_x = (
        gx.amplitude * np.cos(angle)
        - gy.amplitude * np.sin(angle)
    )

    expected_y = (
        gx.amplitude * np.sin(angle)
        + gy.amplitude * np.cos(angle)
    )

    assert (
        by_channel["x"].type
        == "grad"
    )

    assert (
        by_channel["y"].type
        == "grad"
    )

    np.testing.assert_allclose(
        np.max(
            by_channel[
                "x"
            ].waveform
        ),
        expected_x,
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        np.max(
            by_channel[
                "y"
            ].waveform
        ),
        expected_y,
        rtol=1e-6,
        atol=1e-6,
    )
def test_oblique_gradient_safety_scale():
    rotation = (
        planning_euler_to_matrix(
            20.0,
            30.0,
            40.0,
        )
    )

    (
        mixing_bound,
        limit_scale,
    ) = logical_gradient_safety_scale(
        rotation
    )

    assert mixing_bound > 1.0
    assert limit_scale < 1.0

    physical_bound = (
        mixing_bound
        * limit_scale
    )

    np.testing.assert_allclose(
        physical_bound,
        0.95,
        atol=1e-12,
    )
def test_axis_aligned_gradient_safety_scale():
    (
        mixing_bound,
        limit_scale,
    ) = logical_gradient_safety_scale(
        np.eye(3)
    )

    np.testing.assert_allclose(
        mixing_bound,
        1.0,
    )

    np.testing.assert_allclose(
        limit_scale,
        1.0,
    )
def test_add_gradients_preserves_triangle_fall():
    system = _make_system()

    # A perfectly legal triangle.
    #
    # Its slew is only 40% of the physical
    # system limit.
    triangle_rise_time = 40e-6

    triangle_amplitude = (
        0.40
        * system.max_slew
        * triangle_rise_time
    )

    triangle = pp.make_trapezoid(
        channel="x",
        amplitude=triangle_amplitude,
        rise_time=triangle_rise_time,
        flat_time=0.0,
        system=system,
    )

    # A longer gradient forces add_gradients()
    # to rasterize and zero-pad both events.
    companion = pp.make_trapezoid(
        channel="x",
        amplitude=(
            0.01
            * system.max_slew
            * 20e-6
        ),
        rise_time=20e-6,
        flat_time=60e-6,
        system=system,
    )

    combined = add_gradients(
        grads=[
            triangle,
            companion,
        ],
        system=system,
    )

    slew = (
        np.diff(
            combined.waveform
        )
        / system.grad_raster_time
    )

    assert (
        np.max(
            np.abs(slew)
        )
        < system.max_slew
    )