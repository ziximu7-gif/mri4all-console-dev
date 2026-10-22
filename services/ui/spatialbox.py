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

    allow_outside_reference: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    def clamp(self):
        import math

        values = (
            self.center_x,
            self.center_y,
            self.center_z,
            self.size_x,
            self.size_y,
            self.size_z,
            self.rotation_x,
            self.rotation_y,
            self.rotation_z,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite planning geometry")

        minimum = (
            1e-6
            if self.allow_outside_reference
            else 0.02
        )

        maximum = (
            float("inf")
            if self.allow_outside_reference
            else 1.0
        )

        # 1. Restrict box sizes.
        for axis in "xyz":
            attr = f"size_{axis}"
            value = getattr(self, attr)

            setattr(
                self,
                attr,
                max(minimum, min(maximum, value)),
            )

        # 2. Restrict centers only for legacy boxes.
        if not self.allow_outside_reference:
            for axis in "xyz":
                size = getattr(self, f"size_{axis}")
                attr = f"center_{axis}"
                center = getattr(self, attr)

                setattr(
                    self,
                    attr,
                    max(
                        size / 2.0,
                        min(
                            1.0 - size / 2.0,
                            center,
                        ),
                    ),
                )

        # 3. Normalize rotation angles.
        for axis in "xyz":
            attr = f"rotation_{axis}"
            angle = getattr(self, attr)

            setattr(
                self,
                attr,
                (angle + 180.0) % 360.0 - 180.0,
            )

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


@dataclass
class PlanningRegion:
    id: str
    state: PlanningState
    editable: bool = True


@dataclass
class PlanningScene:
    # Physical size of the reference coordinate cube.
    reference_fov_m: float

    # Stable source Localizer provenance.
    source_localizer_id: str

    # One region per next Localizer acquisition.
    regions: dict[str, PlanningRegion] = field(
        default_factory=dict
    )

    active_id: str = "Axial"

    reset_boxes: dict[str, dict] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def select(self, region_id: str):
        if region_id not in self.regions:
            raise KeyError(region_id)
        self.active_id = region_id

    @property
    def active_region(self) -> PlanningRegion:
        return self.regions[self.active_id]

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "coordinate_system": "scanner_xyz_v1",
            "source_localizer_id": self.source_localizer_id,
            "reference_fov_mm": [
                self.reference_fov_m * 1000.0
            ] * 3,
            "boxes": {
                name: region.state.fov_box.as_dict()
                for name, region in self.regions.items()
            },
        }