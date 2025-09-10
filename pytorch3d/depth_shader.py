from typing import Optional

import torch
import torch.nn as nn
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import BlendParams, hard_rgb_blend
from pytorch3d.structures.utils import padded_to_packed


class DepthShader(nn.Module):
    """No light shader."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        """Initlialize without blend_params."""
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, cameras, **kwargs) -> torch.Tensor:
        """Sample without light."""
        world_to_view_transform = cameras.get_world_to_view_transform()
        verts_depth = world_to_view_transform.transform_points(
            meshes.verts_padded())[..., 2:3]
        faces = meshes.faces_packed()  # (F, 3)
        verts_depth = padded_to_packed(verts_depth)
        faces_depth = verts_depth[faces]
        depth_map = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_depth)
        return depth_map[..., 0, :]