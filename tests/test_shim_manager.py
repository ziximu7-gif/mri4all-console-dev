from common.types import ScanTask
from sequences.common.pydanticConfig import (
    configCreator,
)

import services.shim.shim_manager as shim_manager
import pytest

def test_process_b0_shim_updates_config_but_not_snapshot(
    monkeypatch,
):
    config_data = configCreator()

    config_data.shim_parameters.shim_x = 0.10
    config_data.shim_parameters.shim_y = -0.04
    config_data.shim_parameters.shim_z = 0.03

    written = {}

    monkeypatch.setattr(
        shim_manager,
        "reading_json_parameter",
        lambda: config_data,
    )

    def fake_write(config_data):
        written["config"] = config_data

    monkeypatch.setattr(
        shim_manager,
        "writing_json_parameter",
        fake_write,
    )

    monkeypatch.setattr(
        shim_manager,
        "calculate_shim_correction",
        lambda b0_fit: {
            "shim_x": -0.02,
            "shim_y": 0.01,
            "shim_z": -0.005,
        },
    )

    task = ScanTask()

    # Snapshot of shim used during B0 acquisition.
    task.adjustment.shim.shim_x = 0.10
    task.adjustment.shim.shim_y = -0.04
    task.adjustment.shim.shim_z = 0.03

    task.other["b0_fit"] = {
        "gradient_x_hz_per_m": 20.0,
        "gradient_y_hz_per_m": -10.0,
        "gradient_z_hz_per_m": 5.0,
    }

    result = shim_manager.process_b0_shim(
        task
    )

    assert result["after"]["shim_x"] == pytest.approx(
        0.08,
        abs=1e-12,
    )
    assert result["after"]["shim_y"] == pytest.approx(
        -0.03,
        abs=1e-12,
    )
    assert result["after"]["shim_z"] == pytest.approx(
        0.025,
        abs=1e-12,
    )

    # Very important:
    # acquisition snapshot must remain unchanged.
    assert task.adjustment.shim.shim_x == 0.10
    assert task.adjustment.shim.shim_y == -0.04
    assert task.adjustment.shim.shim_z == 0.03

    persisted = written["config"]

    assert (
        persisted.shim_parameters.shim_x
        == pytest.approx(0.08, abs=1e-12)
    )
    assert (
        persisted.shim_parameters.shim_y
        == pytest.approx(-0.03, abs=1e-12)
    )
    assert (
        persisted.shim_parameters.shim_z
        == pytest.approx(0.025, abs=1e-12)
    )


def test_process_b0_shim_rejects_stale_snapshot(
    monkeypatch,
):
    config_data = configCreator()

    # Current scanner state has changed.
    config_data.shim_parameters.shim_x = 0.20

    monkeypatch.setattr(
        shim_manager,
        "reading_json_parameter",
        lambda: config_data,
    )

    monkeypatch.setattr(
        shim_manager,
        "calculate_shim_correction",
        lambda b0_fit: {
            "shim_x": 0.0,
            "shim_y": 0.0,
            "shim_z": 0.0,
        },
    )

    task = ScanTask()

    # B0 was acquired at a different shim.
    task.adjustment.shim.shim_x = 0.10

    task.other["b0_fit"] = {}

    with pytest.raises(
        RuntimeError,
        match="Scanner shim changed",
    ):
        shim_manager.process_b0_shim(
            task
        )