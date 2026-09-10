import numpy as np
import pypulseq as pp

from sequences.common.gradient_transform import (
    transform_gradient_events,
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