import glob
try:
    import sip  # type: ignore
except ImportError:
    from PyQt5 import sip  # type: ignore
import pickle
from pathlib import Path
from typing import Literal, Optional
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *  # type: ignore

import pyqtgraph as pg  # type: ignore
import pydicom
import numpy as np
from PyQt5 import QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import common.logger as logger
from common.types import ResultTypes, ScanTask, TimeSeriesResult
from services.ui.spatialbox import PlanningState
from common.geometry import (
    orientation_plane_axes,
    planning_euler_to_matrix,
    planning_matrix_to_euler,
)

log = logger.get_logger()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use("dark_background")
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class StaticTextItem(pg.TextItem):
    """
    Stays where you put it and ignores viewport translation/zoom.
    """

    def updateTransform(self, force=False):
        if not self.isVisible():
            return

        p = self.parentItem()
        if p is None:
            pt = QtGui.QTransform()
        else:
            pt = p.sceneTransform()

        if not force and pt == self._lastTransform:
            return
        self.setTransform(pt.inverted()[0])
        self._lastTransform = pt
        self.updateTextPos()


class ViewerWidget(QWidget):

    planning_changed = pyqtSignal()
    # layout: QBoxLayout
    widget: Optional[QWidget] = None
    viewed_scan_task: Optional[ScanTask] = None
    viewer_mode: ResultTypes = "empty"

    def __init__(self):
        super(ViewerWidget, self).__init__()

        self.setLayout(
            QVBoxLayout(self)
        )

        self.layout().setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.layout().setSpacing(0)

        # -------------------------------------------------
        # Scan planning
        # -------------------------------------------------

        self.planning_state = None
        self.planning_orientation = None

        self.fov_roi = None
        self.shim_roi = None
        self.fov_label = None
        self.shim_label = None

        # Prevent recursive updates when one viewer
        # updates another viewer.
        self.updating_planning_rois = False
        # State of the ROI currently being manipulated.
        #
        # This lets us distinguish:
        #   move
        #   resize
        #   rotate
        #
        # instead of rewriting all Box3D properties
        # for every sigRegionChanged event.
        self._planning_interactions = {}

        self.set_empty_viewer()

        # def __del__(self):
        #     self.clear_view()

    def clear_view(self):
        if self.widget:
            widget_to_delete = self.widget
            self.layout().removeWidget(self.widget)
            sip.delete(widget_to_delete)
            self.widget = None
            self.viewed_scan_task = None
        self.fov_roi = None
        self.shim_roi = None
        self.fov_label = None
        self.shim_label = None

        self._planning_interactions.clear()

        self.viewer_mode = "empty"

    def set_empty_viewer(self):
        self.widget = QWidget()
        self.widget.setStyleSheet("background-color: #000;")
        self.layout().addWidget(self.widget)

    def view_data(self, file_path: str, viewer_mode: ResultTypes, task) -> bool:
        """
        Used to load results into the viewer for the inline widgets
        """
        self.clear_view()
        self.viewer_mode = viewer_mode
        if viewer_mode == "dicom":
            self.load_dicoms(file_path, task)
            return True
        elif viewer_mode == "plot":
            self.load_pickled_plot(file_path, task)
            return True
        else:
            self.set_empty_viewer()
            self.viewer_mode = "empty"
            return False

    def set_planning_context(
        self,
        orientation,
        planning_state,
    ):
        """
        Tell this viewer which localizer plane it represents
        and which shared PlanningState it should use.
        """

        self.planning_orientation = orientation
        self.planning_state = planning_state

    def _plane_axes(self):
        """
        Return horizontal and vertical normalized axes.

        Prototype convention:

            Axial:
                horizontal = X
                vertical   = Y

            Coronal:
                horizontal = X
                vertical   = Z

            Sagittal:
                horizontal = Y
                vertical   = Z
        """
        if self.planning_orientation is None:
            return None, None

        try:
            return orientation_plane_axes(
                self.planning_orientation
            )
        except ValueError:
            return None, None
            
    def _projection_components(
        self,
        box,
    ):
        """
        Return the matrix that maps the
        3D box-local sizes into the visible
        width/height of this viewer.

        projected_size =
            size_matrix @ [sx, sy, sz]
        """

        rotation = (
            planning_euler_to_matrix(
                box.rotation_x,
                box.rotation_y,
                box.rotation_z,
            )
        )

        if (
            self.planning_orientation
            == "Axial"
        ):
            plane_axes = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )

            primary_axis = 0

        elif (
            self.planning_orientation
            == "Coronal"
        ):
            plane_axes = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            primary_axis = 0

        elif (
            self.planning_orientation
            == "Sagittal"
        ):
            plane_axes = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            primary_axis = 1

        else:
            return None

        projected_axes = (
            plane_axes
            @ rotation
        )

        sizes = np.array(
            [
                box.size_x,
                box.size_y,
                box.size_z,
            ],
            dtype=float,
        )

        primary = (
            projected_axes[
                :,
                primary_axis,
            ]
        )

        primary_norm = (
            np.linalg.norm(
                primary
            )
        )

        if primary_norm < 1e-9:
            projected_lengths = (
                np.linalg.norm(
                    projected_axes,
                    axis=0,
                )
                * sizes
            )

            primary_axis = int(
                np.argmax(
                    projected_lengths
                )
            )

            primary = (
                projected_axes[
                    :,
                    primary_axis,
                ]
            )

            primary_norm = (
                np.linalg.norm(
                    primary
                )
            )

        if primary_norm < 1e-9:
            return None

        horizontal_direction = (
            primary
            / primary_norm
        )

        # Keep a stable angle representation.
        if (
            horizontal_direction[0]
            < 0
        ):
            horizontal_direction = (
                -horizontal_direction
            )

        vertical_direction = np.array(
            [
                -horizontal_direction[1],
                horizontal_direction[0],
            ]
        )

        size_matrix = np.vstack(
            [
                np.abs(
                    horizontal_direction
                    @ projected_axes
                ),
                np.abs(
                    vertical_direction
                    @ projected_axes
                ),
            ]
        )

        angle = np.rad2deg(
            np.arctan2(
                horizontal_direction[1],
                horizontal_direction[0],
            )
        )

        return (
            size_matrix,
            float(angle),
        )


    def _projected_box_geometry(
        self,
        box,
    ):
        """
        Project the rotated 3D Box3D into
        this localizer plane.
        """

        projection = (
            self._projection_components(
                box
            )
        )

        if projection is None:
            return (
                0.0,
                0.0,
                0.0,
            )

        (
            size_matrix,
            angle,
        ) = projection

        sizes = np.array(
            [
                box.size_x,
                box.size_y,
                box.size_z,
            ],
            dtype=float,
        )

        projected_size = (
            size_matrix
            @ sizes
        )

        return (
            float(projected_size[0]),
            float(projected_size[1]),
            angle,
        )


    def _box_angle(
        self,
        box,
    ):
        _, _, angle = (
            self._projected_box_geometry(
                box
            )
        )

        return angle

    def _planning_image_size(self):

        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            return None

        image = self.widget.image

        if image is None:
            return None

        if len(image.shape) < 2:
            return None

        height = image.shape[-2]
        width = image.shape[-1]

        return float(width), float(height)

    def _box_to_rect(
        self,
        box,
    ):
        """
        Project Box3D onto this viewer plane.

        Returns:
            x, y, width, height
        in image pixel coordinates.
        """

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return None

        image_width, image_height = (
            image_size
        )

        horizontal_axis, vertical_axis = (
            self._plane_axes()
        )

        if horizontal_axis is None:
            return None

        horizontal_center = getattr(
            box,
            f"center_{horizontal_axis}",
        )

        vertical_center = getattr(
            box,
            f"center_{vertical_axis}",
        )

        (
            horizontal_size,
            vertical_size,
            _,
        ) = self._projected_box_geometry(box)

        roi_width = (
            horizontal_size
            * image_width
        )

        roi_height = (
            vertical_size
            * image_height
        )

        roi_x = (
            horizontal_center
            * image_width
            - roi_width / 2.0
        )

        roi_y = (
            vertical_center
            * image_height
            - roi_height / 2.0
        )

        return (
            roi_x,
            roi_y,
            roi_width,
            roi_height,
        )
    def _roi_geometry_normalized(
        self,
        roi,
    ):
        """
        Read ROI center, size and angle.

        Center and size are returned in
        normalized localizer coordinates.
        """

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return None

        (
            image_width,
            image_height,
        ) = image_size

        if (
            image_width <= 0
            or image_height <= 0
        ):
            return None

        size = roi.size()

        roi_width = float(
            size.x()
        )

        roi_height = float(
            size.y()
        )

        local_center = QPointF(
            roi_width / 2.0,
            roi_height / 2.0,
        )

        parent_center = (
            roi.mapToParent(
                local_center
            )
        )

        center = np.array(
            [
                float(
                    parent_center.x()
                )
                / image_width,

                float(
                    parent_center.y()
                )
                / image_height,
            ],
            dtype=float,
        )

        normalized_size = np.array(
            [
                roi_width
                / image_width,

                roi_height
                / image_height,
            ],
            dtype=float,
        )

        return (
            center,
            normalized_size,
            float(
                roi.angle()
            ),
        )


    def _begin_planning_roi_interaction(
        self,
        roi,
        box,
    ):
        if self.updating_planning_rois:
            return

        geometry = (
            self._roi_geometry_normalized(
                roi
            )
        )

        if geometry is None:
            return

        (
            _,
            roi_size,
            roi_angle,
        ) = geometry

        self._planning_interactions[
            id(roi)
        ] = {
            "last_size": (
                roi_size.copy()
            ),
            "last_angle": (
                roi_angle
            ),
            "rotation_matrix": (
                planning_euler_to_matrix(
                    box.rotation_x,
                    box.rotation_y,
                    box.rotation_z,
                )
            ),
        }


    def _update_box_center_from_roi(
        self,
        box,
        center,
    ):
        (
            horizontal_axis,
            vertical_axis,
        ) = self._plane_axes()

        if horizontal_axis is None:
            return

        setattr(
            box,
            f"center_{horizontal_axis}",
            float(center[0]),
        )

        setattr(
            box,
            f"center_{vertical_axis}",
            float(center[1]),
        )


    def _update_box_size_from_roi(
        self,
        box,
        target_size,
    ):
        """
        Resize the 3D box while preserving
        the current 3D rotation.

        A 2D projection provides two size
        constraints for three Box3D sizes,
        so use the minimum-change solution.
        """

        projection = (
            self._projection_components(
                box
            )
        )

        if projection is None:
            return

        (
            size_matrix,
            _,
        ) = projection

        current_sizes = np.array(
            [
                box.size_x,
                box.size_y,
                box.size_z,
            ],
            dtype=float,
        )

        current_projection = (
            size_matrix
            @ current_sizes
        )

        projection_error = (
            target_size
            - current_projection
        )

        # Minimum-norm size correction:
        #
        #     A * ds = requested 2D change
        #
        # This automatically gives the intuitive
        # result at zero rotation:
        #
        # Axial    -> X / Y
        # Coronal  -> X / Z
        # Sagittal -> Y / Z
        delta_sizes = (
            np.linalg.pinv(
                size_matrix,
                rcond=1e-6,
            )
            @ projection_error
        )

        new_sizes = (
            current_sizes
            + delta_sizes
        )

        if not np.all(
            np.isfinite(
                new_sizes
            )
        ):
            return

        new_sizes = np.clip(
            new_sizes,
            0.02,
            1.0,
        )

        box.size_x = float(
            new_sizes[0]
        )

        box.size_y = float(
            new_sizes[1]
        )

        box.size_z = float(
            new_sizes[2]
        )


    def _plane_delta_rotation(
        self,
        angle_delta,
    ):
        """
        Build a scanner-space rotation
        corresponding to an in-plane mouse
        rotation in this viewer.
        """

        if (
            self.planning_orientation
            == "Axial"
        ):
            return (
                planning_euler_to_matrix(
                    0.0,
                    0.0,
                    angle_delta,
                )
            )

        if (
            self.planning_orientation
            == "Coronal"
        ):
            return (
                planning_euler_to_matrix(
                    0.0,
                    angle_delta,
                    0.0,
                )
            )

        if (
            self.planning_orientation
            == "Sagittal"
        ):
            return (
                planning_euler_to_matrix(
                    angle_delta,
                    0.0,
                    0.0,
                )
            )

        return np.eye(
            3,
            dtype=float,
        )


    def _update_box_rotation_from_roi(
        self,
        box,
        interaction,
        current_angle,
    ):
        previous_angle = float(
            interaction[
                "last_angle"
            ]
        )

        angle_delta = (
            (
                current_angle
                - previous_angle
                + 180.0
            )
            % 360.0
            - 180.0
        )

        if abs(angle_delta) < 1e-4:
            return

        delta_rotation = (
            self._plane_delta_rotation(
                angle_delta
            )
        )

        current_rotation = (
            interaction[
                "rotation_matrix"
            ]
        )

        # In-plane rotation is around a
        # scanner-space axis, therefore
        # pre-multiply the existing local-to-
        # scanner rotation.
        new_rotation = (
            delta_rotation
            @ current_rotation
        )

        (
            rotation_x,
            rotation_y,
            rotation_z,
        ) = planning_matrix_to_euler(
            new_rotation
        )

        box.rotation_x = (
            rotation_x
        )

        box.rotation_y = (
            rotation_y
        )

        box.rotation_z = (
            rotation_z
        )

        interaction[
            "rotation_matrix"
        ] = new_rotation


    def _roi_to_box(
        self,
        roi,
        box,
    ):
        """
        Update Box3D according to the actual
        type of ROI manipulation.

        Move:
            center only

        Resize:
            center + size

        Rotate:
            center + rotation

        Size and rotation are never modified
        simultaneously merely because
        sigRegionChanged was emitted.
        """

        geometry = (
            self._roi_geometry_normalized(
                roi
            )
        )

        if geometry is None:
            return

        (
            center,
            current_size,
            current_angle,
        ) = geometry

        interaction = (
            self._planning_interactions.get(
                id(roi)
            )
        )

        # Safety fallback if PyQtGraph emits
        # Changed without Started.
        if interaction is None:
            self._begin_planning_roi_interaction(
                roi,
                box,
            )

            interaction = (
                self._planning_interactions.get(
                    id(roi)
                )
            )

        if interaction is None:
            return

        previous_size = (
            interaction[
                "last_size"
            ]
        )

        previous_angle = float(
            interaction[
                "last_angle"
            ]
        )

        angle_delta = (
            (
                current_angle
                - previous_angle
                + 180.0
            )
            % 360.0
            - 180.0
        )

        size_changed = (
            np.max(
                np.abs(
                    current_size
                    - previous_size
                )
            )
            > 1e-6
        )

        angle_changed = (
            abs(angle_delta)
            > 1e-4
        )

        # Translation is valid for all
        # interaction types.
        self._update_box_center_from_roi(
            box,
            center,
        )

        if angle_changed:
            # ROTATE:
            # preserve Box3D size.
            self._update_box_rotation_from_roi(
                box,
                interaction,
                current_angle,
            )

        elif size_changed:
            # RESIZE:
            # preserve Box3D rotation.
            self._update_box_size_from_roi(
                box,
                current_size,
            )

        # Otherwise this was just MOVE:
        # center has already been updated.

        box.clamp()

        interaction[
            "last_size"
        ] = current_size.copy()

        interaction[
            "last_angle"
        ] = current_angle


    def _finish_planning_roi_interaction(
        self,
        roi,
        box,
    ):
        self._planning_interactions.pop(
            id(roi),
            None,
        )

        # At mouse release, make the active
        # ROI exactly match the resolved Box3D.
        #
        # Do NOT do this during every mouse move.
        self.updating_planning_rois = True

        try:
            self._apply_box_to_roi(
                roi,
                box,
            )

        finally:
            self.updating_planning_rois = False

        self._update_planning_labels()

    def _apply_box_to_roi(
        self,
        roi,
        box,
    ):
        """
        Project Box3D into this viewer and apply:

            center
            size
            rotation

        to an existing PyQtGraph ROI.
        """

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return

        image_width, image_height = (
            image_size
        )

        horizontal_axis, vertical_axis = (
            self._plane_axes()
        )

        if horizontal_axis is None:
            return

        center_horizontal = getattr(
            box,
            f"center_{horizontal_axis}",
        )

        center_vertical = getattr(
            box,
            f"center_{vertical_axis}",
        )

        (
            size_horizontal,
            size_vertical,
            angle,
        ) = self._projected_box_geometry(box)

        roi_width = (
            size_horizontal
            * image_width
        )

        roi_height = (
            size_vertical
            * image_height
        )

        desired_center_x = (
            center_horizontal
            * image_width
        )

        desired_center_y = (
            center_vertical
            * image_height
        )
        # Apply size first
        roi.setSize(
            [
                roi_width,
                roi_height,
            ],
            update=False,
        )

        # Reset origin temporarily
        roi.setPos(
            [
                0.0,
                0.0,
            ],
            update=False,
        )

        # Apply rotation
        roi.setAngle(
            angle,
            update=False,
        )

        # Find where the local center currently appears
        local_center = QPointF(
            roi_width / 2.0,
            roi_height / 2.0,
        )

        current_center = (
            roi.mapToParent(
                local_center
            )
        )

        # Translate so the ROI center is exactly
        # the desired Box3D center.
        roi.setPos(
            [
                desired_center_x
                - float(
                    current_center.x()
                ),

                desired_center_y
                - float(
                    current_center.y()
                ),
            ],
            update=False,
        )

        roi.stateChanged(
            finish=False
        )



    def create_planning_rois(self):
        """
        Create the FOV and Shim rectangles
        on the current DICOM viewer.
        """

        if self.planning_state is None:
            return

        if self.planning_orientation is None:
            return

        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            return

        # Avoid creating twice.
        if self.fov_roi is not None:
            return

        # -------------------------------------------------
        # FOV box
        # -------------------------------------------------

        fov_rect = self._box_to_rect(
            self.planning_state.fov_box
        )

        if fov_rect is None:
            return

        (
            fov_x,
            fov_y,
            fov_width,
            fov_height,
        ) = fov_rect
        self.fov_roi = pg.RectROI(
            [
                fov_x,
                fov_y,
            ],
            [
                fov_width,
                fov_height,
            ],
            angle=self._box_angle(
                self.planning_state.fov_box
            ),
            pen=pg.mkPen(
                "y",
                width=3,
            ),
            movable=True,
            rotatable=True,
            resizable=True,
        )

        self.widget.getView().addItem(
            self.fov_roi
        )
        self.fov_roi.addRotateHandle(
            [1.0, 0.0],
            [0.5, 0.5],
        )
        self.fov_roi.addScaleHandle(
            [1.0, 1.0],
            [0.0, 0.0],
        )

        self.fov_roi.addScaleHandle(
            [0.0, 0.0],
            [1.0, 1.0],
        )
        self.fov_label = pg.TextItem(
            text="FOV",
            color="y",
            anchor=(0, 1),
        )

        self.widget.getView().addItem(
            self.fov_label
        )

        # -------------------------------------------------
        # Shim box
        # -------------------------------------------------

        shim_rect = self._box_to_rect(
            self.planning_state.shim_box
        )

        (
            shim_x,
            shim_y,
            shim_width,
            shim_height,
        ) = shim_rect

        self.shim_roi = pg.RectROI(
            [
                shim_x,
                shim_y,
            ],
            [
                shim_width,
                shim_height,
            ],
            angle=self._box_angle(
                self.planning_state.shim_box
            ),
            pen=pg.mkPen(
                "c",
                width=2,
            ),
            movable=True,
            rotatable=True,
            resizable=True,
        )

        self.widget.getView().addItem(
            self.shim_roi
        )
        self.shim_roi.addRotateHandle(
            [1.0, 0.0],
            [0.5, 0.5],
        )
        self.shim_roi.addScaleHandle(
            [1.0, 1.0],
            [0.0, 0.0],
        )

        self.shim_roi.addScaleHandle(
            [0.0, 0.0],
            [1.0, 1.0],
        )
        self.shim_label = pg.TextItem(
            text="Shim",
            color="c",
            anchor=(0, 1),
        )

        self.widget.getView().addItem(
            self.shim_label
        )

        # -------------------------------------------------
        # Connect interaction signals
        # -------------------------------------------------

        self.fov_roi.sigRegionChangeStarted.connect(
            self._fov_roi_change_started
        )

        self.fov_roi.sigRegionChanged.connect(
            self._fov_roi_changed
        )

        self.fov_roi.sigRegionChangeFinished.connect(
            self._fov_roi_change_finished
        )


        self.shim_roi.sigRegionChangeStarted.connect(
            self._shim_roi_change_started
        )

        self.shim_roi.sigRegionChanged.connect(
            self._shim_roi_changed
        )

        self.shim_roi.sigRegionChangeFinished.connect(
            self._shim_roi_change_finished
        )
        self._apply_box_to_roi(
            self.fov_roi,
            self.planning_state.fov_box,
        )

        self._apply_box_to_roi(
            self.shim_roi,
            self.planning_state.shim_box,
        )

        self._update_planning_labels()

    def _update_planning_labels(self):
        """
        Keep FOV / Shim labels attached to each ROI
        and display the current in-plane angle.
        """

        if (
            self.fov_roi is not None
            and
            self.fov_label is not None
        ):

            fov_pos = (
                self.fov_roi.pos()
            )

            fov_angle = (
                self.fov_roi.angle()
            )

            self.fov_label.setText(
                f"FOV  {fov_angle:.1f} deg"
            )

            self.fov_label.setPos(
                float(fov_pos.x()),
                float(fov_pos.y()),
            )

        if (
            self.shim_roi is not None
            and
            self.shim_label is not None
        ):

            shim_pos = (
                self.shim_roi.pos()
            )

            shim_angle = (
                self.shim_roi.angle()
            )

            self.shim_label.setText(
                f"Shim  {shim_angle:.1f} deg"
            )

            self.shim_label.setPos(
                float(shim_pos.x()),
                float(shim_pos.y()),
            )

    def _fov_roi_change_started(
        self,
        *args,
    ):
        if (
            self.fov_roi is None
            or self.planning_state is None
        ):
            return

        self._begin_planning_roi_interaction(
            self.fov_roi,
            self.planning_state.fov_box,
        )


    def _fov_roi_changed(
        self,
        *args,
    ):
        if self.updating_planning_rois:
            return

        if (
            self.planning_state is None
            or self.fov_roi is None
        ):
            return

        self._roi_to_box(
            self.fov_roi,
            self.planning_state.fov_box,
        )

        self._update_planning_labels()

        self.planning_changed.emit()


    def _fov_roi_change_finished(
        self,
        *args,
    ):
        if (
            self.fov_roi is None
            or self.planning_state is None
        ):
            return

        self._finish_planning_roi_interaction(
            self.fov_roi,
            self.planning_state.fov_box,
        )

        self.planning_changed.emit()


    def _shim_roi_change_started(
        self,
        *args,
    ):
        if (
            self.shim_roi is None
            or self.planning_state is None
        ):
            return

        self._begin_planning_roi_interaction(
            self.shim_roi,
            self.planning_state.shim_box,
        )


    def _shim_roi_changed(
        self,
        *args,
    ):
        if self.updating_planning_rois:
            return

        if (
            self.planning_state is None
            or self.shim_roi is None
        ):
            return

        self._roi_to_box(
            self.shim_roi,
            self.planning_state.shim_box,
        )

        self._update_planning_labels()

        self.planning_changed.emit()


    def _shim_roi_change_finished(
        self,
        *args,
    ):
        if (
            self.shim_roi is None
            or self.planning_state is None
        ):
            return

        self._finish_planning_roi_interaction(
            self.shim_roi,
            self.planning_state.shim_box,
        )

        self.planning_changed.emit()

    def refresh_planning_rois(self):
        """
        Refresh this viewer from the shared PlanningState.

        Called when another localizer viewer changes
        FOV or Shim geometry.
        """

        if self.planning_state is None:
            return

        if self.planning_orientation is None:
            return

        if self.fov_roi is None:
            return

        if self.shim_roi is None:
            return

        self.updating_planning_rois = True

        try:

            self._apply_box_to_roi(
                self.fov_roi,
                self.planning_state.fov_box,
            )

            self._apply_box_to_roi(
                self.shim_roi,
                self.planning_state.shim_box,
            )

        finally:

            self.updating_planning_rois = False

        # Put it HERE.
        self._update_planning_labels()

    def clear_planning_context(self):
        self.planning_orientation = None
        self.planning_state = None

        self.fov_roi = None
        self.shim_roi = None

    def load_dicoms(self, input_path, task: Optional[ScanTask] = None):
        if not input_path:
            self.set_empty_viewer()
            return

        lstFilesDCM = None
        if isinstance(input_path, list):
            lstFilesDCM = input_path
        else:
            path = Path(input_path)
            if not path.is_file():
                lstFilesDCM = [str(name) for name in glob.glob(input_path + "*.dcm")]
            else:
                lstFilesDCM = [input_path]
        lstFilesDCM.sort()
        if len(lstFilesDCM) < 1:
            self.set_empty_viewer()
            return

        ds = pydicom.dcmread(lstFilesDCM[0])
        ConstPixelDims = (len(lstFilesDCM), int(ds.Rows), int(ds.Columns))
        ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)
        for filenameDCM in lstFilesDCM:
            ds = pydicom.dcmread(filenameDCM)
            ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

        self.widget = pg.ImageView()
        self.widget.setImage(ArrayDicom)
        self.widget.timeLine.setPen(color=(200, 200, 200), width=8)
        self.widget.timeLine.setHoverPen(color=(255, 255, 255), width=8)

        # viewer_widget.ui.histogram.hide()
        self.widget.ui.roiBtn.hide()
        self.widget.ui.menuBtn.hide()
        self.widget.autoRange()

        if task:
            text = StaticTextItem(
                html=f"""<span style='font-size: 16px; color: #999;'>
                    {task.patient.last_name}, {task.patient.first_name}<br/>
                    {task.patient.mrn}<br/>
                    {task.protocol_name}<br/>
                    Scan {task.scan_number}<br/>
                    </span><br/>""",
                anchor=(0, 0),
            )
            text.setPos(0, 0)  # todo: this only works with 0,0 position
            self.widget.addItem(text)

        self.layout().addWidget(self.widget)
        if (
            self.planning_state is not None
            and
            self.planning_orientation is not None
        ):
            self.create_planning_rois()

    def load_pickled_plot(self, input_path, task: Optional[ScanTask] = None):
        if not input_path:
            self.set_empty_viewer()
            return

        pickled_file_path = Path(input_path)

        if not pickled_file_path.is_file():
            log.error("File not found: " + str(pickled_file_path))
            return

        # TODO: Add error handling!
        self.widget = QWidget()
        self.widget.setLayout(QVBoxLayout(self.widget))
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        plt.style.use("dark_background")
        with open(pickled_file_path, "rb") as pickle_file:
            fig = pickle.load(pickle_file)

        figCanvas = FigureCanvasQTAgg(fig)
        fig.tight_layout()

        toolbar = NavigationToolbar2QT(figCanvas, self)
        toolbar.setStyleSheet(
            "QToolBar::separator { background-color: #0C1123; } QFrame, QFrame:hover { border: 0px solid #000; }  QToolBar { background-color: #000; } QToolButton { background-color: #262C44; } QToolButton:checked { background-color: #FFF; }  QToolButton:disabled { background-color: #000; } QToolButton:hover { background-color: #E0A526; }"
        )
        unwanted_buttons = ["Back", "Forward"]
        for x in toolbar.actions():
            if x.text() in unwanted_buttons:
                toolbar.removeAction(x)

        self.widget.layout().addWidget(figCanvas)
        self.widget.layout().addWidget(toolbar)
        self.layout().addWidget(self.widget)

        if task:
            figCanvas.setToolTip(
                f"{task.scan_number}:  {task.protocol_name}"
            )

        # -------------------------------------------------
        # Interactive span selector for plot results
        # -------------------------------------------------

        # List storing the axis most recently clicked
        curr_ax = []

        axis = fig.get_axes()

        self.textvar = None

        # Detect the currently selected matplotlib axis
        def on_click(event):
            if event.inaxes:
                curr_ax[:] = [
                    event.inaxes
                ]

        # Apply the selected x-range to all plot axes
        def onselect(xmin, xmax):

            # Ignore an accidental click without dragging
            if xmin == xmax:

                for ax, span in zip(
                    axis,
                    list_of_spans,
                ):
                    span.set_visible(
                        False
                    )

                if self.textvar:
                    self.textvar.remove()
                    self.textvar = None

                fig.canvas.draw_idle()

                return

            # Display the same selected range
            # on all sequence plot axes
            for ax, span in zip(
                axis,
                list_of_spans,
            ):
                span.set_visible(
                    True
                )

                span.extents = (
                    xmin,
                    xmax,
                )

            txt = (
                f"start = {xmin:.2f}, "
                f"end = {xmax:.2f}, "
                f"delta = {xmax - xmin:.2f}"
            )

            if self.textvar:
                self.textvar.remove()

            self.textvar = plt.figtext(
                0.5,
                0.01,
                txt,
                wrap=True,
                horizontalalignment="center",
                fontsize=10,
            )

            fig.canvas.draw_idle()

        # Create one horizontal SpanSelector
        # for every matplotlib axis in the figure
        list_of_spans = [
            SpanSelector(
                ax,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(
                    alpha=0.5,
                    facecolor="#262C44",
                ),
                interactive=True,
                drag_from_anywhere=True,
            )
            for ax in axis
        ]

        figCanvas.mpl_connect(
            "button_press_event",
            on_click,
        )

    def layoutUpdate(self):
        if self.viewer_mode == "plot":
            self.widget.layout().itemAt(0).widget().figure.tight_layout()
            self.widget.layout().itemAt(0).widget().figure.canvas.draw()


# def load_plot(self, result: Optional[TimeSeriesResult] = None):
#     sc = MplCanvas(self)
#     if result is None:
#         result = TimeSeriesResult(data=np.random.normal(size=10).tolist())

#     result.show(sc.axes)
#     self.layout().addWidget(sc)
#     self.widget = sc
