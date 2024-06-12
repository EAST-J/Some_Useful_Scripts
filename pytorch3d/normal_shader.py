# The code is adapted from: https://github.com/smplbody/hmr-benchmarks/blob/6d782f06af1333cd45f63cce7a6440391fc28ab1/mmhuman3d/core/visualization/renderer/torch3d_renderer/shader.py
from typing import Optional

import torch
import torch.nn as nn
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import BlendParams, hard_rgb_blend
from pytorch3d.structures.utils import padded_to_packed


class NormalShader(nn.Module):
    """No light shader."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        """Initlialize without blend_params."""
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, cameras=None) -> torch.Tensor:
        """Sample without light."""
        if cameras is None:
            verts_normal = meshes.verts_normals_padded()
        else:
            world_to_view_transform = cameras.get_world_to_view_transform()
            verts_normal = world_to_view_transform.transform_normals(
                meshes.verts_normals_padded())
        faces = meshes.faces_packed()  # (F, 3)
        verts_normal = padded_to_packed(verts_normal)
        faces_normal = verts_normal[faces]
        normal_map = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_normal)
        return normal_map[..., 0, :]
    
