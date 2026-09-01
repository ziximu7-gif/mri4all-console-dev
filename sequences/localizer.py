import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import uic

import pypulseq as pp  # type: ignore
import external.seq.adjustments_acq.config as cfg
from external.seq.adjustments_acq.scripts import run_pulseq

from sequences import PulseqSequence
from sequences.common import make_se_2D
from sequences.common import view_traj
from sequences.common import view_sequence
import common.logger as logger
from common.types import ResultItem
import common.config as config

log = logger.get_logger()


class SequenceSE_2D(PulseqSequence, registry_key=Path(__file__).stem):
    # Sequence parameters
    param_TE: int = 20
    param_TR: int = 3000
    param_NSA: int = 1
    param_FOV: int = 20
    param_Orientation: str = "Axial"
    param_Base_Resolution: int = 96
    param_BW: int = 32000
    param_Trajectory: str = "Catisian"
    param_PE_Ordering: str = "Center_out"
    param_PF: int = 1
    param_view_traj: bool = True

    @classmethod
    def get_readable_name(self) -> str:
        return "Localizer"

    def setup_ui(self, widget) -> bool:
        seq_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(f"{seq_path}/{self.get_name()}/interface.ui", widget)

        # Localizer always acquires all 3 orientations.
        # Hide the old single-orientation selector.
        widget.Orientation_ComboBox.hide()
        widget.label_7.hide()

        return True

    def get_parameters(self) -> dict:
        return {
            "TE": self.param_TE,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "FOV": self.param_FOV,
            "Orientation": self.param_Orientation,
            "Base_Resolution": self.param_Base_Resolution,
            "BW": self.param_BW,
            "Trajectory": self.param_Trajectory,
            "PE_Ordering": self.param_PE_Ordering,
            "PF": self.param_PF,
            "view_traj": self.param_view_traj,
        }

    @classmethod
    def get_default_parameters(self) -> dict:
        return {
            "TE": 20,
            "TR": 3000,
            "NSA": 1,
            "FOV": 20,
            "Orientation": "Axial",
            "Base_Resolution": 96,
            "BW": 32000,
            "Trajectory": "Cartesian",
            "PE_Ordering": "Center_out",
            "PF": 1,
            "view_traj": True,
        }

    def set_parameters(self, parameters, scan_task) -> bool:
        self.problem_list = []
        try:
            self.param_TE = parameters["TE"]
            self.param_TR = parameters["TR"]
            self.param_NSA = parameters["NSA"]
            self.param_FOV = parameters["FOV"]
            self.param_Orientation = parameters["Orientation"]
            self.param_Base_Resolution = parameters["Base_Resolution"]
            self.param_BW = parameters["BW"]
            self.param_Trajectory = parameters["Trajectory"]
            self.param_PE_Ordering = parameters["PE_Ordering"]
            self.param_PF = parameters["PF"]
            self.param_view_traj = parameters["view_traj"]
        except:
            self.problem_list.append("Invalid parameters provided")
            return False
        return self.validate_parameters(scan_task)

    def write_parameters_to_ui(self, widget) -> bool:
        widget.TESpinBox.setValue(self.param_TE)
        widget.TRSpinBox.setValue(self.param_TR)
        widget.NSA_SpinBox.setValue(self.param_NSA)
        widget.Orientation_ComboBox.setCurrentText(self.param_Orientation)
        widget.FOV_SpinBox.setValue(self.param_FOV)
        widget.Base_Resolution_SpinBox.setValue(self.param_Base_Resolution)
        widget.BW_SpinBox.setValue(self.param_BW)
        widget.Trajectory_ComboBox.setCurrentText(self.param_Trajectory)
        widget.PE_Ordering_ComboBox.setCurrentText(self.param_PE_Ordering)
        widget.PF_SpinBox.setValue(self.param_PF)
        widget.visualize_traj_CheckBox.setCheckState(self.param_view_traj)

        return True

    def read_parameters_from_ui(self, widget, scan_task) -> bool:
        self.problem_list = []
        self.param_TE = widget.TESpinBox.value()
        self.param_TR = widget.TRSpinBox.value()
        self.param_NSA = widget.NSA_SpinBox.value()
        self.param_Orientation = widget.Orientation_ComboBox.currentText()
        self.param_FOV = widget.FOV_SpinBox.value()
        self.param_Base_Resolution = widget.Base_Resolution_SpinBox.value()
        self.param_BW = widget.BW_SpinBox.value()
        self.param_Trajectory = widget.Trajectory_ComboBox.currentText()
        self.param_PE_Ordering = widget.PE_Ordering_ComboBox.currentText()
        self.param_PF = widget.PF_SpinBox.value()
        self.param_view_traj = widget.visualize_traj_CheckBox.isChecked()
        self.validate_parameters(scan_task)
        return self.is_valid()

    def validate_parameters(self, scan_task) -> bool:
        if self.param_TE > self.param_TR:
            self.problem_list.append("TE cannot be longer than TR")
        return self.is_valid()

    def calculate_sequence(self, scan_task) -> bool:
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"
        log.info("Calculating sequence " + self.get_name())
        scan_task.processing.recon_mode = "localizer2d"
        scan_task.processing.dim = 2
        scan_task.processing.dim_size = (
            f"{self.param_Base_Resolution},"
            f"{self.param_Base_Resolution}"
        )
        scan_task.processing.oversampling_read = 0
        orientations = [
            "Axial",
            "Coronal",
            "Sagittal",
        ]
        self.seq_files = []
        for index, orientation in enumerate(orientations):
            seq_file = (
                self.get_working_folder()
                + f"/seq/localizer_{index}_{orientation}.seq"
            )
            make_se_2D.pypulseq_se2D(
                inputs={
                    "TE": self.param_TE,
                    "TR": self.param_TR,
                    "NSA": 1,
                    "FOV": self.param_FOV,
                    "Orientation": orientation,
                    "Base_Resolution": self.param_Base_Resolution,
                    "BW": self.param_BW,
                    "Trajectory": "Cartesian",
                    "PE_Ordering": "Center_out",
                    "PF": 1,
                    "view_traj": False,
                },
                check_timing=True,
                output_file=seq_file,
                output_folder=self.get_working_folder(),
            )

            self.seq_files.append((orientation, seq_file))
        # elif self.Trajectory == "Radial":
        # pypulseq_se2D_radial(
        #    inputs={"TE": self.param_TE, "TR": self.param_TR}, check_timing=True, output_file=self.seq_file_path
        # )
    # ---------------------------------------------------------
    # Generate Pulseq sequence visualization
    # ---------------------------------------------------------

        if self.param_view_traj:

            log.info(
                "Generating Localizer sequence visualization"
            )

            sequence_sources = [
                seq_file
                for orientation, seq_file
                in self.seq_files
            ]

            sequence_prefixes = [
                orientation.lower()
                for orientation, seq_file
                in self.seq_files
            ]

            visualization_results = (
                view_sequence.visualize_sequences(
                    sequence_sources=
                        sequence_sources,
                    prefixes=
                        sequence_prefixes,
                    output_folder=(
                        self.get_working_folder()
                        + "/other"
                    ),
                    time_range=(
                        0,
                        float("inf"),
                    ),
                    time_disp="ms",
                    plot_type="Gradient",
                )
            )

            for visualization_result in (
                visualization_results
            ):

                log.info(
                    "Generated sequence visualization: "
                    + str(
                        visualization_result
                    )
                )

            log.info(
                "Done generating Localizer "
                "sequence visualization"
            )

        # Register all Pulseq plots as MRI4ALL results
        # -------------------------------------------------

            for (
                orientation_data,
                visualization_result,
            ) in zip(
                self.seq_files,
                visualization_results,
            ):

                orientation = (
                    orientation_data[0]
                )

                # =============================================
                # Pulseq Figure 1:
                # ADC / RF magnitude / RF-ADC phase
                # =============================================

                rf_result = ResultItem()

                rf_result.name = (
                    f"{orientation} Sequence - RF / ADC"
                )

                rf_result.description = (
                    f"{orientation} Pulseq visualization: "
                    "ADC, RF magnitude and RF/ADC phase"
                )

                rf_result.type = "plot"

                rf_result.primary = False

                # Only auto-open one sequence plot initially.
                # All others will be selectable in the GUI.
                if orientation == "Axial":
                    rf_result.autoload_viewer = 4
                else:
                    rf_result.autoload_viewer = 0

                rf_result.file_path = (
                    f"other/"
                    f"{orientation.lower()}_rf_adc.plot"
                )

                scan_task.results.append(
                    rf_result
                )

                # =============================================
                # Pulseq Figure 2:
                # Gx / Gy / Gz
                # =============================================

                gradient_result = ResultItem()

                gradient_result.name = (
                    f"{orientation} Sequence - Gradients"
                )

                gradient_result.description = (
                    f"{orientation} Pulseq visualization: "
                    "Gx, Gy and Gz"
                )

                gradient_result.type = "plot"

                gradient_result.primary = False

                gradient_result.autoload_viewer = 0

                gradient_result.file_path = (
                    f"other/"
                    f"{orientation.lower()}_gradients.plot"
                )

                scan_task.results.append(
                    gradient_result
                )

        log.info("Done calculating sequence " + self.get_name())
        self.calculated = True

        return True

    def run_sequence(self, scan_task) -> bool:
        log.info("Running localizer sequence")

        raw_data = {}

        for orientation, seq_file in self.seq_files:

            log.info(f"Running localizer group: {orientation}")

            rxd, rx_t = run_pulseq(
                seq_file=seq_file,
                rf_center=cfg.LARMOR_FREQ,
                tx_t=1,
                grad_t=10,
                tx_warmup=100,
                shim_x=0,
                shim_y=0,
                shim_z=0,
                grad_cal=False,
                save_np=False,
                save_mat=False,
                save_msgs=False,
                gui_test=False,
                case_path=self.get_working_folder(),
                hardware_simulation=config.get_config().is_hardware_simulation(),
            )

            raw_data[orientation] = rxd

            log.info(
                f"Finished localizer group {orientation}, "
                f"samples={len(rxd)}"
            )

    # -------------------------------------------------
    # Save each group separately
    # -------------------------------------------------

        raw_folder = self.get_working_folder() + "/rawdata"

        for orientation, rxd in raw_data.items():

            filename = (
                raw_folder
                + f"/localizer_{orientation.lower()}.npy"
            )

            np.save(filename, rxd)

            log.info(
                f"Saved {orientation} raw data to {filename}"
            )

        log.info("Done running localizer sequence")

        return True
