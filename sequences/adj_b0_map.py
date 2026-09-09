from pathlib import Path
from common.geometry import cm_to_m
import external.seq.adjustments_acq.config as cfg
from external.seq.adjustments_acq.scripts import run_pulseq

import common.logger as logger

from sequences import PulseqSequence
from sequences.common import make_gre_3D


log = logger.get_logger()


class B0MapAdjustment(
    PulseqSequence,
    registry_key=Path(__file__).stem,
):
    param_TE1: float = 10.0
    param_TE2: float = 15.0
    param_TR: float = 100.0
    param_NSA: int = 1

    param_FOV: float = 20.0
    param_baseresolution: int = 32
    param_slices: int = 8

    param_BW: int = 32000
    param_FA: float = 20.0
    param_ordering: str = "linear_up"
    param_dummy_shots: int = 10

    @classmethod
    def get_readable_name(cls) -> str:
        return "B0 Map and Shim"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Dual-echo 3D GRE acquisition "
            "for B0 field mapping."
        )

    def setup_ui(self, widget) -> bool:
        # No dedicated UI yet.
        return True

    def get_parameters(self) -> dict:
        return {
            "TE1": self.param_TE1,
            "TE2": self.param_TE2,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "FOV": self.param_FOV,
            "baseresolution": (
                self.param_baseresolution
            ),
            "slices": self.param_slices,
            "BW": self.param_BW,
            "FA": self.param_FA,
            "ordering": self.param_ordering,
        }

    @classmethod
    def get_default_parameters(cls) -> dict:
        return {
            "TE1": 10.0,
            "TE2": 15.0,
            "TR": 100.0,
            "NSA": 1,
            "FOV": 20.0,
            "baseresolution": 32,
            "slices": 8,
            "BW": 32000,
            "FA": 20.0,
            "ordering": "linear_up",
        }

    def set_parameters(
        self,
        parameters,
        scan_task,
    ) -> bool:
        self.problem_list = []

        try:
            self.param_TE1 = parameters["TE1"]
            self.param_TE2 = parameters["TE2"]
            self.param_TR = parameters["TR"]
            self.param_NSA = parameters["NSA"]

            self.param_FOV = parameters["FOV"]
            self.param_baseresolution = (
                parameters["baseresolution"]
            )
            self.param_slices = parameters["slices"]

            self.param_BW = parameters["BW"]
            self.param_FA = parameters["FA"]
            self.param_ordering = (
                parameters["ordering"]
            )

        except (KeyError, TypeError, ValueError):
            self.problem_list.append(
                "Invalid parameters provided"
            )
            return False

        return self.validate_parameters(
            scan_task
        )

    def write_parameters_to_ui(
        self,
        widget,
    ) -> bool:
        return True

    def read_parameters_from_ui(
        self,
        widget,
        scan_task,
    ) -> bool:
        return self.validate_parameters(
            scan_task
        )

    def validate_parameters(
        self,
        scan_task,
    ) -> bool:
        if self.param_TE1 <= 0:
            self.problem_list.append(
                "TE1 must be greater than zero"
            )

        if self.param_TE2 <= self.param_TE1:
            self.problem_list.append(
                "TE2 must be greater than TE1"
            )

        if self.param_TR <= self.param_TE2:
            self.problem_list.append(
                "TR must be greater than TE2"
            )

        return self.is_valid()

    def calculate_sequence(
        self,
        scan_task,
    ) -> bool:
        log.info(
            "Calculating sequence "
            + self.get_name()
        )

        scan_task.processing.recon_mode = (
            "b0_map"
        )

        scan_task.processing.dim = 3

        scan_task.processing.dim_size = (
            f"{self.param_slices},"
            f"{self.param_baseresolution},"
            f"{2 * self.param_baseresolution}"
        )

        scan_task.processing.oversampling_read = 2
        base_fov_m = cm_to_m(
            self.param_FOV
        )

        scan_task.other["b0_geometry"] = {
            "fov_x_m": base_fov_m,
            "fov_y_m": base_fov_m,
            "fov_z_m": base_fov_m / 2.0,
        }

        # Make sure reconstruction has access
        # to the actual echo times used.
        scan_task.parameters["TE1"] = (
            self.param_TE1
        )
        scan_task.parameters["TE2"] = (
            self.param_TE2
        )

        self.seq_file_path = (
            self.get_working_folder()
            + "/seq/acq0.seq"
        )

        success = make_gre_3D.pypulseq_gre3D(
            inputs={
                "TE": self.param_TE1,
                "echo_times": [
                    self.param_TE1,
                    self.param_TE2,
                ],
                "TR": self.param_TR,
                "NSA": self.param_NSA,
                "orientation": "Axial",
                "FOV": self.param_FOV,
                "baseresolution": (
                    self.param_baseresolution
                ),
                "slices": self.param_slices,
                "BW": self.param_BW,
                "ordering": self.param_ordering,
                "FA": self.param_FA,
                "dummy_shots": (
                    self.param_dummy_shots
                ),
            },
            check_timing=True,
            output_file=self.seq_file_path,
            working_folder=(
                self.get_working_folder()
            ),
        )

        if not success:
            log.error(
                "Unable to calculate sequence "
                + self.get_name()
            )
            return False

        self.calculated = True

        log.info(
            "Done calculating sequence "
            + self.get_name()
        )

        return True

    def run_sequence(
        self,
        scan_task,
    ) -> bool:
        log.info(
            "Running sequence "
            + self.get_name()
        )

        expected_duration_sec = int(
            self.param_TR
            * (
                self.param_baseresolution
                * self.param_slices
                + self.param_dummy_shots
            )
            / 1000
        )

        run_pulseq(
            seq_file=self.seq_file_path,
            rf_center=cfg.LARMOR_FREQ,
            tx_t=1,
            grad_t=10,
            tx_warmup=100,
            shim_x=cfg.SHIM_X,
            shim_y=cfg.SHIM_Y,
            shim_z=cfg.SHIM_Z,
            grad_cal=False,
            save_np=True,
            save_mat=False,
            save_msgs=False,
            gui_test=False,
            case_path=self.get_working_folder(),
            raw_filename="raw",
            expected_duration_sec=(
                expected_duration_sec
            ),
            plot_instructions=False,
        )

        scan_task.adjustment.rf.larmor_frequency = (
            cfg.LARMOR_FREQ
        )

        log.info(
            "Done running sequence "
            + self.get_name()
        )

        return True