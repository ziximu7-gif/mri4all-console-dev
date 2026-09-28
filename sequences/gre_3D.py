import os
from pathlib import Path
import copy
import datetime

import numpy as np
from PyQt5 import uic

from common.simulation import (
    GRE3DSimulationContext,
)

import external.seq.adjustments_acq.config as cfg
from external.seq.adjustments_acq.scripts import (
    run_pulseq,
)

from sequences import PulseqSequence
from sequences.common import (
    make_gre_3D,
    view_sequence,
)

from common.constants import *
import common.logger as logger
from common.types import ResultItem
import common.helper as helper
import common.config as config

from sequences.common.planning import (
    resolve_task_planning,
)

from common.geometry import (
    cm_to_m,
    mm_to_m,
    orientation_encoding_matrix,
    planning_matrix_to_euler,
)


log = logger.get_logger()

class SequenceGRE_3D(PulseqSequence, registry_key=Path(__file__).stem):
    # Sequence parameters
    param_TE: int = 50
    param_TR: int = 250
    param_NSA: int = 1
    param_orientation: str = "Axial"
    param_FOV: int = 15
    param_baseresolution: int = 64
    param_slices: int = 8
    param_BW: int = 32000
    param_trajectory: str = "Cartesian"
    param_ordering: str = "linear_up"
    param_dummy_shots: int = 20
    param_readout_direction: str = ("Horizontal")

    # UI-only reference to the currently edited ScanTask.
    #
    # Not persistent state. Used only to derive the
    # read-only geometry display in the parameter UI.
    _ui_scan_task = None

    @classmethod
    def get_readable_name(self) -> str:
        return "3D Gradient Echo"

    @classmethod
    def get_description(self) -> str:
        return "Volumetric 3D GRE acquisition with Cartesian sampling"

    def setup_ui(self, widget) -> bool:
        seq_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(f"{seq_path}/{self.get_name()}/interface.ui", widget)

        widget.TR_SpinBox.valueChanged.connect(self.update_info)
        widget.FOV_SpinBox.valueChanged.connect(self.update_info)
        widget.Baseresolution_SpinBox.valueChanged.connect(self.update_info)
        widget.Slices_SpinBox.valueChanged.connect(self.update_info)
        widget.Orientation_ComboBox.currentTextChanged.connect(self.update_info)
        widget.ReadoutDirection_ComboBox.currentTextChanged.connect(
            self.update_info
        )

        widget.PlanningFOVRead_Value.valueChanged.connect(
            self._planned_fov_ui_changed
        )

        widget.PlanningFOVPhase_Value.valueChanged.connect(
            self._planned_fov_ui_changed
        )

        widget.PlanningSlabThickness_Value.valueChanged.connect(
            self._planned_fov_ui_changed
        )
        return True

    def _current_geometry_ui_values(
        self,
    ):
        """
        Resolve the geometry that would actually be used by
        GRE acquisition for the current parameter UI state.

        Planning geometry takes precedence over the legacy
        scalar FOV parameter.
        """

        widget = self.main_widget

        orientation = (
            widget
            .Orientation_ComboBox
            .currentText()
        )

        readout_direction = (
            widget
            .ReadoutDirection_ComboBox
            .currentText()
        )

        scan_task = getattr(
            self,
            "_ui_scan_task",
            None,
        )

        planned_geometry = None
        planned_encoding = None

        if scan_task is not None:

            try:
                (
                    planned_geometry,
                    planned_encoding,
                ) = resolve_task_planning(
                    scan_task=scan_task,
                    orientation=orientation,
                    readout_direction=(
                        readout_direction
                    ),
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:

                log.warning(
                    "Unable to resolve planned GRE geometry "
                    "for parameter UI: "
                    + str(exc)
                )

        if (
            planned_geometry is not None
            and planned_encoding is not None
        ):

            fov_m = np.asarray(
                planned_encoding.fov_logical_m,
                dtype=float,
            )

            center_mm = (
                np.asarray(
                    planned_geometry.center_scanner_m,
                    dtype=float,
                )
                * 1000.0
            )

            rotation_deg = np.asarray(
                planning_matrix_to_euler(
                    planned_geometry
                    .rotation_local_to_scanner
                ),
                dtype=float,
            )

            planning_active = True

        else:

            # Legacy GRE behavior:
            #
            # read  = FOV
            # phase = FOV
            # third = FOV / 2
            base_fov_m = float(
                cm_to_m(
                    widget.FOV_SpinBox.value()
                )
            )

            fov_m = np.array(
                [
                    base_fov_m,
                    base_fov_m,
                    0.5 * base_fov_m,
                ],
                dtype=float,
            )

            center_mm = None
            rotation_deg = None
            planning_active = False

        base_resolution = max(
            1,
            int(
                widget
                .Baseresolution_SpinBox
                .value()
            ),
        )

        partitions = max(
            1,
            int(
                widget
                .Slices_SpinBox
                .value()
            ),
        )

        voxel_mm = (
            fov_m
            * 1000.0
            / np.array(
                [
                    base_resolution,
                    base_resolution,
                    partitions,
                ],
                dtype=float,
            )
        )

        phase_percent = (
            100.0
            * float(fov_m[1])
            / float(fov_m[0])
        )

        return {
            "planning_active": (
                planning_active
            ),
            "fov_m": fov_m,
            "center_mm": center_mm,
            "rotation_deg": rotation_deg,
            "phase_percent": (
                phase_percent
            ),
            "voxel_mm": voxel_mm,
        }

    def refresh_planning_ui(
        self,
    ):
        """
        Refresh read-only physical geometry derived from the
        current ScanTask and sequence encoding parameters.
        """

        widget = self.main_widget

        values = (
            self._current_geometry_ui_values()
        )

        planning_active = bool(
            values[
                "planning_active"
            ]
        )

        fov_m = values[
            "fov_m"
        ]

        voxel_mm = values[
            "voxel_mm"
        ]

        if planning_active:

            widget.PlanningSource_Value.setText(
                "From Localizer Planning"
            )

            center_mm = values[
                "center_mm"
            ]

            widget.PlanningPosition_Value.setText(
                "X "
                f"{float(center_mm[0]):+.1f}"
                "  Y "
                f"{float(center_mm[1]):+.1f}"
                "  Z "
                f"{float(center_mm[2]):+.1f}"
                " mm"
            )

            rotation_deg = values[
                "rotation_deg"
            ]

            widget.PlanningRotation_Value.setText(
                "Rx "
                f"{float(rotation_deg[0]):+.1f}"
                "  Ry "
                f"{float(rotation_deg[1]):+.1f}"
                "  Rz "
                f"{float(rotation_deg[2]):+.1f}"
                " \u00b0"
            )

        else:

            widget.PlanningSource_Value.setText(
                "Legacy sequence geometry"
            )

            widget.PlanningPosition_Value.setText(
                "-"
            )

            widget.PlanningRotation_Value.setText(
                "-"
            )

        widget.FOV_SpinBox.setEnabled(
            not planning_active
        )

        fov_widgets = (
            widget.PlanningFOVRead_Value,
            widget.PlanningFOVPhase_Value,
            widget.PlanningSlabThickness_Value,
        )

        for current_widget in fov_widgets:
            current_widget.blockSignals(
                True
            )

        try:

            widget.PlanningFOVRead_Value.setValue(
                float(fov_m[0])
                * 1000.0
            )

            widget.PlanningFOVPhase_Value.setValue(
                float(
                    values[
                        "phase_percent"
                    ]
                )
            )

            widget.PlanningSlabThickness_Value.setValue(
                float(fov_m[2])
                * 1000.0
            )

        finally:

            for current_widget in fov_widgets:
                current_widget.blockSignals(
                    False
                )

        for current_widget in fov_widgets:
            current_widget.setEnabled(
                planning_active
            )

        widget.PlanningVoxelSize_Value.setText(
            f"{float(voxel_mm[0]):.2f}"
            " \u00d7 "
            f"{float(voxel_mm[1]):.2f}"
            " \u00d7 "
            f"{float(voxel_mm[2]):.2f}"
            " mm"
        )

        return values

    def _planned_fov_ui_changed(
        self,
    ):
        """
        Write editable logical FOV values back into the
        copied planning geometry of the currently edited GRE
        task.

        This modifies the GRE task's geometry copy, not the
        source Localizer PlanningState.
        """

        scan_task = getattr(
            self,
            "_ui_scan_task",
            None,
        )

        if scan_task is None:
            return

        geometry_data = (
            scan_task.other.get(
                "geometry"
            )
        )

        if not isinstance(
            geometry_data,
            dict,
        ):
            return

        fov_box = geometry_data.get(
            "fov_box"
        )

        reference_fov_mm = (
            geometry_data.get(
                "reference_fov_mm"
            )
        )

        if (
            not isinstance(
                fov_box,
                dict,
            )
            or reference_fov_mm is None
        ):
            return

        widget = self.main_widget

        read_fov_m = (
            float(
                widget
                .PlanningFOVRead_Value
                .value()
            )
            / 1000.0
        )

        phase_percent = float(
            widget
            .PlanningFOVPhase_Value
            .value()
        )

        third_fov_m = (
            float(
                widget
                .PlanningSlabThickness_Value
                .value()
            )
            / 1000.0
        )

        phase_fov_m = (
            read_fov_m
            * phase_percent
            / 100.0
        )

        logical_fov_m = np.array(
            [
                read_fov_m,
                phase_fov_m,
                third_fov_m,
            ],
            dtype=float,
        )

        encoding_matrix = (
            orientation_encoding_matrix(
                widget
                .Orientation_ComboBox
                .currentText(),
                readout_direction=(
                    widget
                    .ReadoutDirection_ComboBox
                    .currentText()
                ),
            )
        )

        # Inverse of:
        #
        # logical =
        #     abs(E).T @ box_local
        #
        # E is currently an axis permutation matrix.
        box_fov_m = (
            np.abs(
                encoding_matrix
            )
            @ logical_fov_m
        )

        reference_fov_m = np.asarray(
            mm_to_m(
                reference_fov_mm
            ),
            dtype=float,
        )

        if reference_fov_m.shape != (3,):
            return

        candidate_size_norm = (
            box_fov_m
            / reference_fov_m
        )

        if (
            np.any(
                candidate_size_norm
                < 0.02
            )
            or np.any(
                candidate_size_norm
                > 1.0
            )
        ):
            log.warning(
                "Requested planned FOV is outside "
                "the Localizer reference volume."
            )

            self.refresh_planning_ui()
            return

        center_norm = np.array(
            [
                float(
                    fov_box[
                        "center_x"
                    ]
                ),
                float(
                    fov_box[
                        "center_y"
                    ]
                ),
                float(
                    fov_box[
                        "center_z"
                    ]
                ),
            ],
            dtype=float,
        )

        minimum_center = (
            0.5
            * candidate_size_norm
        )

        maximum_center = (
            1.0
            - 0.5
            * candidate_size_norm
        )

        if (
            np.any(
                center_norm
                < minimum_center
            )
            or np.any(
                center_norm
                > maximum_center
            )
        ):
            log.warning(
                "Requested planned FOV would exceed "
                "the current planning bounds."
            )

            self.refresh_planning_ui()
            return

        fov_box[
            "size_x"
        ] = float(
            candidate_size_norm[0]
        )

        fov_box[
            "size_y"
        ] = float(
            candidate_size_norm[1]
        )

        fov_box[
            "size_z"
        ] = float(
            candidate_size_norm[2]
        )

        self.update_info()

    def merge_ui_other_parameters(
        self,
        other_data,
    ):
        """
        Merge geometry edited by the sequence UI into the
        Other-data document that will be persisted.
        """

        scan_task = getattr(
            self,
            "_ui_scan_task",
            None,
        )

        if scan_task is None:
            return other_data

        geometry = (
            scan_task.other.get(
                "geometry"
            )
        )

        if not isinstance(
            geometry,
            dict,
        ):
            return other_data

        merged = copy.deepcopy(
            other_data
        )

        merged[
            "geometry"
        ] = copy.deepcopy(
            geometry
        )

        return merged

    def update_info(self):

        widget = self.main_widget

        duration_sec = int(
            widget.TR_SpinBox.value()
            * (
                widget
                .Baseresolution_SpinBox
                .value()
                * widget
                .Slices_SpinBox
                .value()
                + self.param_dummy_shots
            )
            / 1000
        )

        duration = str(
            datetime.timedelta(
                seconds=duration_sec
            )
        )

        values = (
            self.refresh_planning_ui()
        )

        voxel_mm = values[
            "voxel_mm"
        ]

        self.show_ui_info_text(
            (
                f"TA: {duration} sec"
                "       "
                "Voxel Size: "
                f"{float(voxel_mm[0]):.2f}"
                " x "
                f"{float(voxel_mm[1]):.2f}"
                " x "
                f"{float(voxel_mm[2]):.2f}"
                " mm"
            )
        )

    def get_parameters(self) -> dict:
        return {
            "TE": self.param_TE,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "orientation": self.param_orientation,
            "FOV": self.param_FOV,
            "baseresolution": self.param_baseresolution,
            "slices": self.param_slices,
            "BW": self.param_BW,
            "trajectory": self.param_trajectory,
            "ordering": self.param_ordering,
            "FA": self.param_FA,
            "readout_direction": (self.param_readout_direction),
        }

    @classmethod
    def get_default_parameters(
        self,
    ) -> dict:
        return {
            "TE": 0,
            "TR": 1000,
            "NSA": 1,
            "orientation": "Axial",
            "FOV": 15,
            "baseresolution": 32,
            "slices": 8,
            "BW": 32000,
            "trajectory": "Cartesian",
            "ordering": "linear_up",
            "FA": 20,
            "readout_direction": ("Horizontal"),
        }

    def set_parameters(self, parameters, scan_task) -> bool:
        self.problem_list = []

        # Keep the currently edited ScanTask only for
        # derived geometry display in the parameter UI.
        #
        # The ScanTask remains the source of planning geometry;
        # no geometry is copied into sequence parameters here.
        self._ui_scan_task = scan_task

        try:
            self.param_TE = parameters["TE"]
            self.param_TR = parameters["TR"]
            self.param_NSA = parameters["NSA"]
            self.param_orientation = parameters["orientation"]
            self.param_FOV = parameters["FOV"]
            self.param_baseresolution = parameters["baseresolution"]
            self.param_slices = parameters["slices"]
            self.param_BW = parameters["BW"]
            self.param_trajectory = parameters["trajectory"]
            self.param_ordering = parameters["ordering"]
            self.param_FA = parameters["FA"]
            self.param_readout_direction = (parameters.get("readout_direction","Horizontal",))
        except:
            self.problem_list.append("Invalid parameters provided")
            return False
        return self.validate_parameters(scan_task)

    def write_parameters_to_ui(self, widget) -> bool:
        widget.TE_SpinBox.setValue(self.param_TE)
        widget.TR_SpinBox.setValue(self.param_TR)
        widget.FA_SpinBox.setValue(self.param_FA)
        widget.NSA_SpinBox.setValue(self.param_NSA)
        widget.Orientation_ComboBox.setCurrentText(self.param_orientation)
        widget.FOV_SpinBox.setValue(self.param_FOV)
        widget.Baseresolution_SpinBox.setValue(self.param_baseresolution)
        widget.Slices_SpinBox.setValue(self.param_slices)
        widget.BW_SpinBox.setValue(self.param_BW)
        widget.Trajectory_ComboBox.setCurrentText(self.param_trajectory)
        widget.Ordering_ComboBox.setCurrentText(self.param_ordering)
        widget.ReadoutDirection_ComboBox.setCurrentText(self.param_readout_direction)

        self.update_info()

        return True

    def read_parameters_from_ui(self, widget, scan_task) -> bool:
        self.problem_list = []
        self._ui_scan_task = scan_task
        self.param_TE = widget.TE_SpinBox.value()
        self.param_TR = widget.TR_SpinBox.value()
        self.param_NSA = widget.NSA_SpinBox.value()
        self.param_orientation = widget.Orientation_ComboBox.currentText()
        self.param_FOV = widget.FOV_SpinBox.value()
        self.param_baseresolution = widget.Baseresolution_SpinBox.value()
        self.param_slices = widget.Slices_SpinBox.value()
        self.param_BW = widget.BW_SpinBox.value()
        self.param_trajectory = widget.Trajectory_ComboBox.currentText()
        self.param_ordering = widget.Ordering_ComboBox.currentText()
        self.param_FA = (widget.FA_SpinBox.value())
        self.param_readout_direction = (widget.ReadoutDirection_ComboBox.currentText())
        self.validate_parameters(scan_task)
        return self.is_valid()

    def validate_parameters(self, scan_task) -> bool:
        if self.param_TE > self.param_TR:
            self.problem_list.append("TE cannot be longer than TR")
        return self.is_valid()

    def calculate_sequence(self, scan_task) -> bool:
        log.info("Calculating sequence " + self.get_name())
        (
            planned_geometry,
            planned_encoding,
        ) = resolve_task_planning(
            scan_task=scan_task,
            orientation=(
                self.param_orientation
            ),
            readout_direction=(
                self.param_readout_direction
            ),
        )
        if planned_geometry is not None:

            scan_task.other[
                "resolved_geometry"
            ] = (
                planned_geometry.as_dict()
            )

            scan_task.other[
                "resolved_encoding"
            ] = (
                planned_encoding.as_dict()
            )

            fov_m = (
                planned_encoding.fov_logical_m
            )

            scan_task.other[
                "geometry_application"
            ] = {
                "fov_size": True,
                "encoding_orientation": (
                    self.param_orientation
                ),
                "readout_direction": (
                    self.param_readout_direction
                ),
                "gradient_transform": True,
                "translation": True,
                "translation_method": (
                    "reconstruction_kspace_phase_ramp"
                ),
                "slab_selection": True,
                "slab_positioning": True,
                "slab_positioning_method": (
                    "rf_waveform_frequency_modulation"
                ),
                "logical_delta_k_1_per_m": [
                    float(1.0 / fov_m[0]),
                    float(1.0 / fov_m[1]),
                    float(1.0 / fov_m[2]),
                ],
            }

            log.info(
                "Resolved planned FOV geometry: "
                + str(
                    scan_task.other[
                        "resolved_geometry"
                    ]
                )
            )

            log.info(
                "Resolved GRE encoding geometry: "
                + str(
                    scan_task.other[
                        "resolved_encoding"
                    ]
                )
            )
            if planned_encoding is not None:
                matrix = (
                    planned_encoding
                    .logical_to_scanner
                )

                log.info(
                    "GRE k-space axes in scanner XYZ: "
                    f"read={matrix[:, 0].tolist()}, "
                    f"phase1={matrix[:, 1].tolist()}, "
                    f"phase2={matrix[:, 2].tolist()}"
                )

        scan_task.processing.recon_mode = "basic3d"
        scan_task.processing.dim = 3
        scan_task.processing.dim_size = f"{self.param_slices},{self.param_baseresolution},{2*self.param_baseresolution}"
        scan_task.processing.oversampling_read = 2
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"

        if not self.generate_pulseq(
            planned_encoding=planned_encoding
        ):
            log.error(
                "Unable to calculate sequence "
                + self.get_name()
            )
            return False

        try:
            self.generate_sequence_visualization(
                scan_task
            )

        except Exception:
            log.exception(
                "Unable to generate GRE "
                "sequence visualization. "
                "Continuing acquisition."
            )

        log.info(
            "Done calculating sequence "
            + self.get_name()
        )

        return True

    def generate_sequence_visualization(
        self,
        scan_task,
    ) -> None:
        """
        Generate Pulseq visualization files for
        the 3D GRE sequence.

        Visualization failure must not prevent
        acquisition or reconstruction.
        """

        output_folder = (
            self.get_working_folder()
            + "/other"
        )

        log.info(
            "Generating full GRE sequence visualization"
        )

        visualization_result = (
            view_sequence.visualize_sequence(
                sequence_source=(
                    self.seq_file_path
                ),
                output_folder=(
                    output_folder
                ),
                prefix="gre3d",
                time_range=(
                    0,
                    float("inf"),
                ),
                time_disp="ms",
                plot_type="Gradient",
            )
        )

        rf_result = ResultItem()
        rf_result.name = (
            "GRE Sequence - RF / ADC"
        )
        rf_result.description = (
            "3D GRE Pulseq RF and ADC "
            "visualization"
        )
        rf_result.type = "plot"
        rf_result.primary = False

        # Flex Viewer
        rf_result.autoload_viewer = 4

        rf_result.file_path = (
            "other/"
            + Path(
                visualization_result[
                    "rf_adc"
                ]
            ).name
        )

        scan_task.results.append(
            rf_result
        )

        gradient_result = ResultItem()
        gradient_result.name = (
            "GRE Sequence - Gradients"
        )
        gradient_result.description = (
            "Physical scanner Gx/Gy/Gz "
            "after FOV rotation"
        )
        gradient_result.type = "plot"
        gradient_result.primary = False

        # Do not overwrite another Viewer
        # automatically.
        gradient_result.autoload_viewer = 0

        gradient_result.file_path = (
            "other/"
            + Path(
                visualization_result[
                    "gradients"
                ]
            ).name
        )

        scan_task.results.append(
            gradient_result
        )

        log.info(
            "GRE sequence visualization "
            "generated successfully: "
            f"{rf_result.file_path}, "
            f"{gradient_result.file_path}"
        )

    def build_simulation_context(
        self,
        scan_task,
    ):
        """
        Adapter layer: extract plain numerical geometry from the
        ScanTask and build the typed GRE3D simulation context.

        The pure simulation layer (common/simulation) must not know
        ScanTask persistence, so all other["resolved_*"] access
        happens here. Returns None when no resolved geometry is
        available (legacy non-planned GRE path falls back to the
        legacy simulation branch in run_pulseq).
        """
        resolved_geometry = scan_task.other.get(
            "resolved_geometry"
        )
        resolved_encoding = scan_task.other.get(
            "resolved_encoding"
        )

        if not isinstance(resolved_geometry, dict):
            return None
        if not isinstance(resolved_encoding, dict):
            return None

        try:
            center_scanner_m = np.asarray(
                resolved_geometry["center_scanner_m"],
                dtype=float,
            )
            center_logical_m = np.asarray(
                resolved_encoding["center_logical_m"],
                dtype=float,
            )
            fov_logical_m = np.asarray(
                resolved_encoding["fov_logical_m"],
                dtype=float,
            )
            logical_to_scanner = np.asarray(
                resolved_encoding["logical_to_scanner"],
                dtype=float,
            )
        except KeyError as exc:
            log.warning(
                "Resolved geometry is missing "
                + str(exc)
                + ". Falling back to legacy simulation."
            )
            return None

        return GRE3DSimulationContext(
            kind="gre3d",
            center_scanner_m=center_scanner_m,
            center_logical_m=center_logical_m,
            fov_logical_m=fov_logical_m,
            logical_to_scanner=(
                logical_to_scanner
            ),
            base_resolution=(
                int(self.param_baseresolution)
            ),
            n_phase=int(self.param_baseresolution),
            n_slice=int(self.param_slices),
            oversampling_read=(
                scan_task.processing.oversampling_read
            ),
        )

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

        hardware_simulation = (
            config.get_config()
            .is_hardware_simulation()
        )

        if hardware_simulation:
            log.info(
                "GRE acquisition mode: "
                "HARDWARE SIMULATION"
            )
        else:
            log.info(
                "GRE acquisition mode: "
                "REAL HARDWARE"
            )

        sim_context = None
        if hardware_simulation:
            sim_context = (
                self.build_simulation_context(
                    scan_task
                )
            )
            if sim_context is not None:
                log.info(
                    "GRE simulation: fixed scanner-space "
                    "phantom with resolved planning geometry"
                )
            else:
                log.info(
                    "GRE simulation: no resolved planning "
                    "geometry; legacy phantom fallback"
                )

        try:
            rxd, rx_t = run_pulseq(
                seq_file=(
                    self.seq_file_path
                ),
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
                case_path=(
                    self.get_working_folder()
                ),
                raw_filename="raw",
                expected_duration_sec=(
                    expected_duration_sec
                ),
                plot_instructions=False,
                hardware_simulation=(
                    hardware_simulation
                ),
                sim_context=sim_context,
            )

        except Exception:
            log.exception(
                "GRE run_pulseq failed."
            )
            return False

        if rxd is None:
            log.error(
                "GRE acquisition returned "
                "no ADC data."
            )
            return False

        log.info(
            "GRE acquisition returned "
            f"{len(rxd)} ADC samples"
        )

        log.info(
            f"GRE rx_t = {rx_t}"
        )

        scan_task.adjustment.rf.larmor_frequency = (
            cfg.LARMOR_FREQ
        )

        log.info(
            "Done running sequence "
            + self.get_name()
        )

        return True

    def generate_pulseq(
        self,
        planned_encoding=None,
    ) -> bool:

        inputs = {
            "TE": self.param_TE,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "orientation": self.param_orientation,
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
        }

        if planned_encoding is not None:
            inputs["planned_center_logical_m"] = (
                planned_encoding
                .center_logical_m
                .tolist()
            )

            inputs["planned_fov_m"] = (
                planned_encoding
                .fov_logical_m
                .tolist()
            )

            inputs[
                "logical_to_scanner"
            ] = (
                planned_encoding
                .logical_to_scanner
                .tolist()
            )


        return make_gre_3D.pypulseq_gre3D(
            inputs=inputs,
            check_timing=True,
            output_file=self.seq_file_path,
            working_folder=(
                self.get_working_folder()
            ),
        )