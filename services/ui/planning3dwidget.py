import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

import pyqtgraph.opengl as gl

from common.geometry import (
    planning_box_to_scan_geometry,
    scan_geometry_box_edges_scanner_m,
)


class Planning3DWidget(QWidget):
    """
    Read-only 3D scanner-space planning viewer.

    Coordinate convention:
        X = scanner X
        Y = scanner Y
        Z = scanner Z

    Scene coordinates are meters, matching common.geometry.

    The widget does not own planning geometry.
    It displays the shared PlanningState supplied by
    ExaminationWindow.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(0)

        self.view = gl.GLViewWidget()

        self.view.setBackgroundColor(
            (0, 0, 0, 255)
        )

        layout.addWidget(
            self.view
        )

        # -------------------------------------------------
        # Shared planning context
        # -------------------------------------------------

        self.planning_state = None

        # Physical size of the centered scanner-space
        # reference volume used by normalized Box3D.
        self.reference_fov_m = None

        # -------------------------------------------------
        # GL scene items
        # -------------------------------------------------

        self.axis_items = []
        self.localizer_plane_items = []

        # One GLLinePlotItem contains all 12 FOV edges.
        #
        # mode="lines" interprets each consecutive pair
        # of points as one independent line segment.
        self.fov_wireframe_item = (
            gl.GLLinePlotItem(
                pos=np.zeros(
                    (24, 3),
                    dtype=float,
                ),
                color=(
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                ),
                width=3,
                antialias=True,
                mode="lines",
            )
        )

        self.fov_wireframe_item.hide()

        self.view.addItem(
            self.fov_wireframe_item
        )

        # Camera looks toward scanner isocenter.
        #
        # Scene units are meters. Current Localizer FOV is
        # around 0.20 m, so 0.45 m gives a useful initial view.
        self.view.setCameraPosition(
            distance=0.45,
            elevation=20,
            azimuth=45,
        )

        self._create_scanner_axes()

    # =====================================================
    # Static scanner reference
    # =====================================================

    def _create_scanner_axes(
        self,
    ):
        """
        Draw positive scanner X/Y/Z axes from isocenter.
        """

        axis_length_m = 0.12

        axis_specs = (
            (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [axis_length_m, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
                (1.0, 0.2, 0.2, 1.0),
            ),
            (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, axis_length_m, 0.0],
                    ],
                    dtype=float,
                ),
                (0.2, 1.0, 0.2, 1.0),
            ),
            (
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, axis_length_m],
                    ],
                    dtype=float,
                ),
                (0.2, 0.5, 1.0, 1.0),
            ),
        )

        for points, color in axis_specs:

            item = gl.GLLinePlotItem(
                pos=points,
                color=color,
                width=3,
                antialias=True,
                mode="lines",
            )

            self.view.addItem(
                item
            )

            self.axis_items.append(
                item
            )

    # =====================================================
    # Planning context
    # =====================================================

    def set_planning_context(
        self,
        planning_state,
        reference_fov_m,
    ):
        """
        Attach the shared PlanningState and physical
        Localizer reference FOV.

        The PlanningState object is shared; this widget does
        not copy it.
        """

        reference_fov_m = float(
            reference_fov_m
        )

        if (
            not np.isfinite(
                reference_fov_m
            )
            or reference_fov_m <= 0.0
        ):
            raise ValueError(
                "Planning 3D reference FOV "
                "must be finite and positive"
            )

        self.planning_state = (
            planning_state
        )

        self.reference_fov_m = (
            reference_fov_m
        )

        self._create_localizer_planes()

        self.refresh_planning_geometry()

    def clear_planning_context(
        self,
    ):
        self.planning_state = None
        self.reference_fov_m = None

        self._clear_localizer_planes()

        self.fov_wireframe_item.hide()

    # =====================================================
    # Localizer planes
    # =====================================================

    def _clear_localizer_planes(
        self,
    ):
        for item in (
            self.localizer_plane_items
        ):
            try:
                self.view.removeItem(
                    item
                )
            except Exception:
                pass

        self.localizer_plane_items = []

    def _create_localizer_planes(
        self,
    ):
        """
        Draw the three centered Localizer center planes.

        Current Localizer geometry:
            Axial    = XY, z=0
            Coronal  = XZ, y=0
            Sagittal = YZ, x=0

        These are center planes only. The acquired 10 mm
        slice thickness is deliberately not rendered here.
        """

        self._clear_localizer_planes()

        if (
            self.reference_fov_m
            is None
        ):
            return

        half = (
            0.5
            * float(
                self.reference_fov_m
            )
        )

        axial = np.array(
            [
                [-half, -half, 0.0],
                [ half, -half, 0.0],
                [ half,  half, 0.0],
                [-half,  half, 0.0],
            ],
            dtype=float,
        )

        coronal = np.array(
            [
                [-half, 0.0, -half],
                [ half, 0.0, -half],
                [ half, 0.0,  half],
                [-half, 0.0,  half],
            ],
            dtype=float,
        )

        sagittal = np.array(
            [
                [0.0, -half, -half],
                [0.0,  half, -half],
                [0.0,  half,  half],
                [0.0, -half,  half],
            ],
            dtype=float,
        )

        plane_specs = (
            (
                axial,
                (
                    0.2,
                    0.5,
                    1.0,
                    0.10,
                ),
                (
                    0.2,
                    0.5,
                    1.0,
                    0.55,
                ),
            ),
            (
                coronal,
                (
                    0.2,
                    1.0,
                    0.5,
                    0.10,
                ),
                (
                    0.2,
                    1.0,
                    0.5,
                    0.55,
                ),
            ),
            (
                sagittal,
                (
                    1.0,
                    0.3,
                    0.7,
                    0.10,
                ),
                (
                    1.0,
                    0.3,
                    0.7,
                    0.55,
                ),
            ),
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=int,
        )

        for (
            vertices,
            face_color,
            edge_color,
        ) in plane_specs:

            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                color=face_color,
                smooth=False,
                drawEdges=False,
                glOptions="translucent",
            )

            outline_points = (
                np.vstack(
                    [
                        vertices,
                        vertices[0],
                    ]
                )
            )

            outline = (
                gl.GLLinePlotItem(
                    pos=outline_points,
                    color=edge_color,
                    width=1,
                    antialias=True,
                    mode="line_strip",
                )
            )

            self.view.addItem(
                mesh
            )

            self.view.addItem(
                outline
            )

            self.localizer_plane_items.extend(
                [
                    mesh,
                    outline,
                ]
            )

    # =====================================================
    # Dynamic FOV
    # =====================================================

    def refresh_planning_geometry(
        self,
    ):
        """
        Rebuild the displayed 3D FOV wireframe from the
        shared PlanningState.

        No geometry is inferred from any 2D ROI.
        """

        if (
            self.planning_state
            is None
        ):
            self.fov_wireframe_item.hide()
            return

        if (
            self.reference_fov_m
            is None
        ):
            self.fov_wireframe_item.hide()
            return

        reference_fov_mm = np.full(
            3,
            float(
                self.reference_fov_m
            )
            * 1000.0,
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

        edges = (
            scan_geometry_box_edges_scanner_m(
                scan_geometry
            )
        )

        # GLLinePlotItem mode="lines":
        #
        #   p0,p1 = edge 0
        #   p2,p3 = edge 1
        #   ...
        #
        # Shape becomes:
        #   (24, 3)
        edge_points = (
            edges.reshape(
                -1,
                3,
            )
        )

        self.fov_wireframe_item.setData(
            pos=edge_points,
            color=(
                1.0,
                1.0,
                0.0,
                1.0,
            ),
            width=3,
            antialias=True,
            mode="lines",
        )

        self.fov_wireframe_item.show()
