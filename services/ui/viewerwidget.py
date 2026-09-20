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
    planning_box_corners_in_encoding_m,
    planning_box_to_scan_geometry,
    intersect_box_with_plane_scanner_m,
    project_box_edges_to_plane_scanner_m,
    cm_to_m,
    localizer_image_plane_geometry,
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
    plot_xlim_changed = pyqtSignal(
        float,
        float,
    )
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

        # Orientation of THIS Localizer viewer
        # (Axial / Coronal / Sagittal).
        self.planning_orientation = None

        # Physical scanner-space center plane of the currently
        # displayed Localizer image.
        self.planning_image_plane = None

        # Physical size of the normalized planning reference
        # volume. Current Localizer v0 uses the same square FOV
        # on all scanner axes.
        self.planning_reference_fov_m = None

        # Pixel data belonging to the currently displayed
        # single-slice Localizer image.
        self.planning_image_array = None

        # Canonical FOV display derived from the shared 3D box:
        # dashed = full wireframe projection
        # solid  = true box / Localizer-plane intersection
        self.fov_projection_item = None
        self.fov_intersection_item = None

        self.fov_basis_items = []

        self.fov_roi = None
        self.shim_roi = None
        self.fov_label = None
        self.shim_label = None

        self.reconstruction_axis_order = None
        self.reconstruction_overlay_items = []

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

        self._updating_plot_xlim = False

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

        # Physical geometry belongs to the currently loaded image.
        self.planning_image_plane = None
        self.planning_reference_fov_m = None
        self.planning_image_array = None

        # Overlay items belonged to the deleted ImageView.
        self.fov_projection_item = None
        self.fov_intersection_item = None

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

    def get_plot_figure(self):
        if self.viewer_mode != "plot":
            return None

        if self.widget is None:
            return None

        layout = self.widget.layout()

        if (
            layout is None
            or layout.count() == 0
        ):
            return None

        canvas = (
            layout.itemAt(0).widget()
        )

        if canvas is None:
            return None

        return getattr(
            canvas,
            "figure",
            None,
        )

    def set_plot_xlim(
        self,
        xmin,
        xmax,
    ):
        fig = self.get_plot_figure()

        if fig is None:
            return

        if self._updating_plot_xlim:
            return

        self._updating_plot_xlim = True

        try:
            for ax in fig.get_axes():
                ax.set_xlim(
                    xmin,
                    xmax,
                )

            fig.canvas.draw_idle()

        finally:
            self._updating_plot_xlim = False

    def _plot_xlim_changed(
        self,
        changed_axis,
    ):
        if self._updating_plot_xlim:
            return

        xmin, xmax = (
            changed_axis.get_xlim()
        )

        self._updating_plot_xlim = True

        try:
            fig = self.get_plot_figure()

            if fig is None:
                return

            for ax in fig.get_axes():
                if ax is changed_axis:
                    continue

                ax.set_xlim(
                    xmin,
                    xmax,
                )

            fig.canvas.draw_idle()

        finally:
            self._updating_plot_xlim = False

        self.plot_xlim_changed.emit(
            float(xmin),
            float(xmax),
        )

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

    def _clear_fov_geometry_overlay(
        self,
    ):
        """
        Remove the canonical solid/dashed FOV display.
        """

        if isinstance(
            self.widget,
            pg.ImageView,
        ):
            view = self.widget.getView()

            for item in (
                self.fov_projection_item,
                self.fov_intersection_item,
            ):
                if item is None:
                    continue

                try:
                    view.removeItem(
                        item
                    )
                except Exception:
                    pass

        self.fov_projection_item = None
        self.fov_intersection_item = None

    def _update_fov_geometry_overlay(
        self,
    ):
        """
        Rebuild the canonical FOV display for this Localizer.

        Solid:
            true intersection between the current Localizer
            center plane and the shared 3D planning FOV.

        Dashed:
            orthographic projection of all twelve 3D FOV
            edges onto the Localizer center plane.

        The RectROI remains an interaction control only.
        It is not the canonical FOV display.
        """

        self._clear_fov_geometry_overlay()

        if self.planning_state is None:
            return

        if self.planning_image_plane is None:
            return

        if self.planning_reference_fov_m is None:
            return

        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            return

        reference_fov_m = float(
            self.planning_reference_fov_m
        )

        if (
            not np.isfinite(reference_fov_m)
            or reference_fov_m <= 0.0
        ):
            return

        # Box3D is stored in normalized scanner X/Y/Z coordinates.
        # Convert it to physical scanner-space geometry.
        reference_fov_mm = np.full(
            3,
            reference_fov_m * 1000.0,
            dtype=float,
        )

        scan_geometry = (
            planning_box_to_scan_geometry(
                fov_box=(
                    self.planning_state
                    .fov_box
                    .as_dict()
                ),
                reference_fov_mm=(
                    reference_fov_mm
                ),
            )
        )

        view = self.widget.getView()

        # -------------------------------------------------
        # Dashed: complete 3D FOV wireframe projection
        # -------------------------------------------------

        projected_edges_scanner = (
            project_box_edges_to_plane_scanner_m(
                scan_geometry,
                self.planning_image_plane,
            )
        )

        projected_edges_xy = (
            self.planning_image_plane
            .scanner_to_image_xy(
                projected_edges_scanner
            )
        )

        projection_x = []
        projection_y = []

        for edge in projected_edges_xy:
            projection_x.extend(
                [
                    float(edge[0, 0]),
                    float(edge[1, 0]),
                    np.nan,
                ]
            )

            projection_y.extend(
                [
                    float(edge[0, 1]),
                    float(edge[1, 1]),
                    np.nan,
                ]
            )

        self.fov_projection_item = (
            pg.PlotDataItem(
                projection_x,
                projection_y,
                connect="finite",
                pen=pg.mkPen(
                    "y",
                    width=2,
                    style=Qt.DashLine,
                ),
            )
        )

        self.fov_projection_item.setZValue(
            20
        )

        view.addItem(
            self.fov_projection_item
        )

        # -------------------------------------------------
        # Solid: true FOV / current Localizer intersection
        # -------------------------------------------------

        intersection_scanner = (
            intersect_box_with_plane_scanner_m(
                scan_geometry,
                self.planning_image_plane,
            )
        )

        # One tangent point has no visible line.
        if intersection_scanner.shape[0] < 2:
            return

        intersection_xy = (
            self.planning_image_plane
            .scanner_to_image_xy(
                intersection_scanner
            )
        )

        # A proper polygon must be explicitly closed.
        if intersection_xy.shape[0] >= 3:
            intersection_xy = np.vstack(
                [
                    intersection_xy,
                    intersection_xy[0],
                ]
            )

        self.fov_intersection_item = (
            pg.PlotDataItem(
                intersection_xy[:, 0],
                intersection_xy[:, 1],
                pen=pg.mkPen(
                    "y",
                    width=3,
                ),
            )
        )

        self.fov_intersection_item.setZValue(
            30
        )

        view.addItem(
            self.fov_intersection_item
        )

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

    def _clear_fov_basis_indicator(self):
        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            self.fov_basis_items = []
            return

        view = self.widget.getView()

        for item in self.fov_basis_items:
            try:
                view.removeItem(item)
            except Exception:
                pass

        self.fov_basis_items = []

    def _update_fov_basis_indicator(self):
        self._clear_fov_basis_indicator()

        if (
            self.planning_state is None
            or self.planning_orientation is None
        ):
            return

        image_size = self._planning_image_size()

        if image_size is None:
            return

        image_width, image_height = image_size

        box = self.planning_state.fov_box

        rotation = planning_euler_to_matrix(
            box.rotation_x,
            box.rotation_y,
            box.rotation_z,
        )

        if self.planning_orientation == "Axial":
            plane_projection = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )

        elif self.planning_orientation == "Coronal":
            plane_projection = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        elif self.planning_orientation == "Sagittal":
            plane_projection = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        else:
            return

        projected_axes = (
            plane_projection
            @ rotation
        )

        horizontal_axis, vertical_axis = (
            self._plane_axes()
        )

        center_x = (
            getattr(
                box,
                f"center_{horizontal_axis}",
            )
            * image_width
        )

        center_y = (
            getattr(
                box,
                f"center_{vertical_axis}",
            )
            * image_height
        )

        indicator_length = (
            0.12
            * min(
                image_width,
                image_height,
            )
        )

        axis_specs = (
            ("X'", 0, "y"),
            ("Y'", 1, "c"),
            ("Z'", 2, "m"),
        )

        view = self.widget.getView()

        for (
            axis_name,
            axis_index,
            color,
        ) in axis_specs:

            direction = (
                projected_axes[
                    :,
                    axis_index,
                ]
            )

            direction_norm = float(
                np.linalg.norm(direction)
            )

            # Axis almost perpendicular to
            # this localizer plane.
            if direction_norm < 0.10:
                continue

            direction = (
                direction
                / direction_norm
            )

            end_x = (
                center_x
                + indicator_length
                * direction[0]
            )

            end_y = (
                center_y
                + indicator_length
                * direction[1]
            )

            line = pg.PlotDataItem(
                [
                    center_x,
                    end_x,
                ],
                [
                    center_y,
                    end_y,
                ],
                pen=pg.mkPen(
                    color,
                    width=3,
                ),
            )

            label = pg.TextItem(
                text=axis_name,
                color=color,
                anchor=(0.5, 0.5),
            )

            label.setPos(
                end_x,
                end_y,
            )

            view.addItem(line)
            view.addItem(label)

            self.fov_basis_items.extend(
                [
                    line,
                    label,
                ]
            )
            
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
                (255, 255, 0, 80),
                width=1,
            ),
            movable=True,
            rotatable=True,
            resizable=True,
        )

        self.widget.getView().addItem(
            self.fov_roi
        )
        self.fov_roi.setZValue(
            40
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

        self._update_fov_basis_indicator()

        self._update_fov_geometry_overlay()

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

        self._update_fov_basis_indicator()

        self._update_fov_geometry_overlay()

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

        self._update_fov_geometry_overlay()

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

        self._update_fov_basis_indicator()

        self._update_fov_geometry_overlay()

    def set_reconstruction_overlay_context(
        self,
        logical_axis_order,
    ):
        """
        Enable read-only acquisition overlays
        for a FOV-native reconstruction.

        logical_axis_order maps the DICOM
        array axes to logical axes:

            0 = read
            1 = phase
            2 = third

        DICOM axis 0 is the row (vertical),
        axis 1 is the column (horizontal).
        """

        self.reconstruction_axis_order = tuple(
            logical_axis_order
        )

    def clear_reconstruction_overlay_context(
        self,
    ):
        self.reconstruction_axis_order = None
        self.reconstruction_overlay_items = []

    def _clear_reconstruction_overlays(self):
        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            self.reconstruction_overlay_items = []
            self.reconstruction_axis_order = None
            return

        view = self.widget.getView()

        for item in self.reconstruction_overlay_items:
            try:
                view.removeItem(item)
            except Exception:
                pass

        self.reconstruction_overlay_items = []

    def create_reconstruction_overlays(
        self,
        task,
    ):
        """
        Draw read-only reconstruction overlay
        items for a FOV-native reconstruction:

            - yellow FOV boundary
            - "Planned FOV" label
            - cyan Shim volume projection
            - axis indicator

        The reconstruction image itself is the
        oblique FOV, so the FOV boundary is
        simply the full image boundary.
        """

        if (
            self.reconstruction_axis_order
            is None
        ):
            return

        if not isinstance(
            self.widget,
            pg.ImageView,
        ):
            return

        image_size = (
            self._planning_image_size()
        )

        if image_size is None:
            return

        image_width, image_height = (
            image_size
        )

        view = self.widget.getView()

        row_axis = (
            self.reconstruction_axis_order[0]
        )

        column_axis = (
            self.reconstruction_axis_order[1]
        )

        # -------------------------------------------------
        # Full-image FOV boundary + label
        # -------------------------------------------------

        fov_outline = pg.PlotDataItem(
            [
                0.0,
                image_width,
                image_width,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                image_height,
                image_height,
                0.0,
            ],
            pen=pg.mkPen(
                "y",
                width=3,
            ),
        )

        view.addItem(
            fov_outline
        )

        self.reconstruction_overlay_items.append(
            fov_outline
        )

        fov_label = pg.TextItem(
            text="Planned FOV",
            color="y",
            anchor=(0, 1),
        )

        fov_label.setPos(
            4.0,
            4.0,
        )

        view.addItem(
            fov_label
        )

        self.reconstruction_overlay_items.append(
            fov_label
        )

        # -------------------------------------------------
        # Shim box projection
        # -------------------------------------------------

        geometry = task.other.get(
            "geometry"
        )

        resolved_geometry = (
            task.other.get(
                "resolved_geometry"
            )
        )

        resolved_encoding = (
            task.other.get(
                "resolved_encoding"
            )
        )

        if (
            isinstance(
                geometry,
                dict,
            )
            and isinstance(
                resolved_geometry,
                dict,
            )
            and isinstance(
                resolved_encoding,
                dict,
            )
        ):

            shim_box = geometry.get(
                "shim_box"
            )

            reference_fov_mm = (
                geometry.get(
                    "reference_fov_mm"
                )
            )

            acquisition_center_scanner_m = (
                resolved_geometry.get(
                    "center_scanner_m"
                )
            )

            logical_to_scanner = (
                resolved_encoding.get(
                    "logical_to_scanner"
                )
            )

            if (
                isinstance(
                    shim_box,
                    dict,
                )
                and reference_fov_mm is not None
                and acquisition_center_scanner_m
                is not None
                and logical_to_scanner is not None
            ):

                try:
                    fov_m = np.asarray(
                        resolved_encoding[
                            "fov_logical_m"
                        ],
                        dtype=float,
                    )

                    shim_corners = (
                        planning_box_corners_in_encoding_m(
                            box=shim_box,
                            reference_fov_mm=(
                                reference_fov_mm
                            ),
                            acquisition_center_scanner_m=(
                                acquisition_center_scanner_m
                            ),
                            logical_to_scanner=(
                                logical_to_scanner
                            ),
                        )
                    )

                    shim_x = (
                        shim_corners[
                            :,
                            column_axis,
                        ]
                        / fov_m[
                            column_axis
                        ]
                        + 0.5
                    ) * image_width

                    shim_y = (
                        shim_corners[
                            :,
                            row_axis,
                        ]
                        / fov_m[
                            row_axis
                        ]
                        + 0.5
                    ) * image_height

                    shim_pixel_corners = (
                        np.column_stack(
                            [
                                shim_x,
                                shim_y,
                            ]
                        )
                    )

                    self._draw_box_projection(
                        shim_pixel_corners,
                        color="c",
                        width=2,
                    )

                except Exception:
                    log.exception(
                        "Unable to draw Shim "
                        "overlay on reconstruction."
                    )

        # -------------------------------------------------
        # Axis indicator
        # -------------------------------------------------

        axis_names = (
            "Read",
            "Phase",
            "Third",
        )

        indicator_row_name = (
            axis_names[row_axis]
        )

        indicator_column_name = (
            axis_names[column_axis]
        )

        through_axis = (
            3
            - row_axis
            - column_axis
        )

        indicator_through_name = (
            axis_names[through_axis]
        )

        axis_indicator = pg.TextItem(
            text=(
                "Vertical:   "
                + indicator_row_name
                + "\n"
                + "Horizontal: "
                + indicator_column_name
                + "\n"
                + "Through:    "
                + indicator_through_name
            ),
            color="#999999",
            anchor=(1, 1),
        )

        axis_indicator.setPos(
            image_width - 4.0,
            image_height - 4.0,
        )

        view.addItem(
            axis_indicator
        )

        self.reconstruction_overlay_items.append(
            axis_indicator
        )

    def _draw_box_projection(
        self,
        corners,
        color,
        width,
    ):
        """
        Draw the 12 edges of an 8-corner 3D box
        in the current 2D view.

        corners must already be expressed in the
        encoding-native pixel coordinate system.
        """

        edges = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),

            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),

            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
        ]

        view = self.widget.getView()

        for start_index, end_index in edges:

            line = pg.PlotDataItem(
                [
                    corners[
                        start_index,
                        0,
                    ],
                    corners[
                        end_index,
                        0,
                    ],
                ],
                [
                    corners[
                        start_index,
                        1,
                    ],
                    corners[
                        end_index,
                        1,
                    ],
                ],
                pen=pg.mkPen(
                    color,
                    width=width,
                ),
            )

            view.addItem(line)

            self.reconstruction_overlay_items.append(
                line
            )

    def get_localizer_plane_image(
        self,
    ):
        """
        Return the currently displayed Localizer image together
        with its explicit scanner-space geometry.

        Returns:
            (orientation, image_array, ImagePlaneGeometry)

        or None when this viewer is not displaying a valid
        Localizer planning image.
        """

        if self.planning_orientation is None:
            return None

        if self.planning_image_plane is None:
            return None

        if self.planning_image_array is None:
            return None

        return (
            self.planning_orientation,
            self.planning_image_array.copy(),
            self.planning_image_plane,
        )

    def clear_planning_context(self):
        self._clear_fov_geometry_overlay()

        self.planning_orientation = None
        self.planning_state = None

        # Plane geometry belongs to the currently loaded image.
        self.planning_image_plane = None
        self.planning_reference_fov_m = None
        self.planning_image_array = None

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

        # -------------------------------------------------
        # Explicit scanner-space center plane of this
        # Localizer image.
        #
        # The physical source of truth is the scan task's
        # planned FOV, NOT the DICOM spatial tags: the
        # current Localizer DICOM spatial tags are not the
        # planning geometry contract.
        # -------------------------------------------------
        self.planning_image_plane = None
        self.planning_reference_fov_m = None

        if (
            task is not None
            and task.sequence == "localizer"
            and self.planning_orientation is not None
            and self.planning_state is not None
        ):
            try:
                fov_cm = float(
                    task.parameters["FOV"]
                )

                if (
                    not np.isfinite(fov_cm)
                    or fov_cm <= 0
                ):
                    raise ValueError(
                        "Invalid Localizer FOV"
                    )

                fov_m = float(
                    cm_to_m(fov_cm)
                )

                self.planning_reference_fov_m = (
                    fov_m
                )

                self.planning_image_plane = (
                    localizer_image_plane_geometry(
                        orientation=(
                            self.planning_orientation
                        ),
                        fov_m=fov_m,
                        rows=int(ds.Rows),
                        columns=int(ds.Columns),
                    )
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                log.warning(
                    "Localizer image plane geometry "
                    "unavailable: missing or invalid "
                    "FOV in scan task parameters"
                )

                self.planning_image_plane = None
                self.planning_reference_fov_m = None

        row_spacing_mm = 1.0
        column_spacing_mm = 1.0

        try:
            pixel_spacing = ds.PixelSpacing

            row_spacing_mm = float(
                pixel_spacing[0]
            )

            column_spacing_mm = float(
                pixel_spacing[1]
            )

            if (
                not np.isfinite(row_spacing_mm)
                or not np.isfinite(column_spacing_mm)
                or row_spacing_mm <= 0
                or column_spacing_mm <= 0
            ):
                raise ValueError(
                    "Invalid DICOM PixelSpacing"
                )

        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
        ):
            row_spacing_mm = 1.0
            column_spacing_mm = 1.0

        ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)
        for filenameDCM in lstFilesDCM:
            ds = pydicom.dcmread(filenameDCM)
            ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

        # -------------------------------------------------
        # Pixel data handed to the 3D planning view.
        #
        # The Localizer planning contract is explicitly
        # single-slice: a multi-slice series is NOT silently
        # reduced to its first slice.
        # -------------------------------------------------
        self.planning_image_array = None

        if (
            task is not None
            and task.sequence == "localizer"
            and self.planning_image_plane is not None
        ):
            if ArrayDicom.shape[0] == 1:
                self.planning_image_array = (
                    np.asarray(
                        ArrayDicom[0],
                    ).copy()
                )
            else:
                log.warning(
                    "Expected single-slice Localizer DICOM, "
                    "but received "
                    + str(ArrayDicom.shape[0])
                    + " slices"
                )

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

        self.widget = pg.ImageView()
        self.widget.setImage(ArrayDicom)
        self.widget.getView().setAspectLocked(
            True,
            ratio=(
                column_spacing_mm
                / row_spacing_mm
            ),
        )
        self.widget.timeLine.setPen(color=(200, 200, 200), width=8)
        self.widget.timeLine.setHoverPen(color=(255, 255, 255), width=8)

        if (
            self.reconstruction_axis_order
            is not None
        ):
            self.create_reconstruction_overlays(
                task
            )

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

        for ax in axis:
            ax.callbacks.connect(
                "xlim_changed",
                self._plot_xlim_changed,
            )

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
