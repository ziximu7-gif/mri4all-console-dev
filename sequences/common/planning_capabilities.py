from dataclasses import dataclass
from typing import Literal

GeometryKind = Literal[
    "none",
    "slice2d",
    "volume3d",
]


@dataclass(frozen=True)
class PlanningCapabilities:
    # Whether this sequence provides reference images.
    provides_reference: bool = False

    # Whether planning requires another scan's images.
    uses_reference_images: bool = False

    # Whether the user may edit the acquisition FOV.
    editable_fov: bool = False

    # Kind of acquisition geometry.
    geometry_kind: GeometryKind = "none"