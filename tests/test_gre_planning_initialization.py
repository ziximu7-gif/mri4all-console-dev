import pytest

from types import SimpleNamespace

from sequences.common.planning import (
    make_initial_gre_geometry,
)


def test_gre_fov_is_independent_from_localizer():
    localizer = SimpleNamespace(
        id="localizer-001",
        sequence="localizer",
        parameters={"FOV": 20},
    )

    gre_parameters = {
        "FOV": 15,
        "orientation": "Axial",
        "readout_direction": "Horizontal",
    }

    geometry = make_initial_gre_geometry(
        gre_parameters,
        localizer,
    )

    box = geometry["fov_box"]

    assert box["size_x"] == pytest.approx(0.75)
    assert box["size_y"] == pytest.approx(0.75)
    assert box["size_z"] == pytest.approx(0.375)

    assert geometry["source_localizer_id"] == (
        "localizer-001"
    )