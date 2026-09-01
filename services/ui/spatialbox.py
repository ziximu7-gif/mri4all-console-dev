from dataclasses import dataclass, field


@dataclass
class Box3D:
    """
    Normalized 3D box.

    All coordinates are currently in range 0.0 ... 1.0.

    This is intentionally NOT millimeters yet because the
    localizer DICOM geometry is not physically correct yet.
    """

    center_x: float = 0.5
    center_y: float = 0.5
    center_z: float = 0.5

    size_x: float = 0.7
    size_y: float = 0.7
    size_z: float = 0.7

    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0

    def clamp(self):
        """
        Keep box inside normalized image volume.
        """

        self.size_x = max(
            0.02,
            min(1.0, self.size_x),
        )

        self.size_y = max(
            0.02,
            min(1.0, self.size_y),
        )

        self.size_z = max(
            0.02,
            min(1.0, self.size_z),
        )

        self.center_x = max(
            self.size_x / 2.0,
            min(
                1.0 - self.size_x / 2.0,
                self.center_x,
            ),
        )

        self.center_y = max(
            self.size_y / 2.0,
            min(
                1.0 - self.size_y / 2.0,
                self.center_y,
            ),
        )

        self.center_z = max(
            self.size_z / 2.0,
            min(
                1.0 - self.size_z / 2.0,
                self.center_z,
            ),
        )

        self.rotation_x = (
            (self.rotation_x + 180.0) % 360.0
        ) - 180.0

        self.rotation_y = (
            (self.rotation_y + 180.0) % 360.0
        ) - 180.0

        self.rotation_z = (
            (self.rotation_z + 180.0) % 360.0
        ) - 180.0

    def as_dict(self):
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
            "rotation_x": self.rotation_x,
            "rotation_y": self.rotation_y,
            "rotation_z": self.rotation_z,
        }


@dataclass
class PlanningState:
    """
    Shared scan-planning state.

    All three localizer viewers reference the SAME object.
    """

    fov_box: Box3D = field(
        default_factory=lambda: Box3D(
            center_x=0.5,
            center_y=0.5,
            center_z=0.5,
            size_x=0.70,
            size_y=0.70,
            size_z=0.70,
        )
    )

    shim_box: Box3D = field(
        default_factory=lambda: Box3D(
            center_x=0.5,
            center_y=0.5,
            center_z=0.5,
            size_x=0.50,
            size_y=0.50,
            size_z=0.50,
        )
    )

    def as_dict(self):
        return {
            "fov_box": self.fov_box.as_dict(),
            "shim_box": self.shim_box.as_dict(),
        }