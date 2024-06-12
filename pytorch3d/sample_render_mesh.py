import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import trimesh



device = "cuda"
obj_info = np.load("./sample_data/0001.npz")
# in open-cv coordinate
R = torch.from_numpy(obj_info["R"].astype(np.float32)).unsqueeze(0)
T = torch.from_numpy(obj_info["T"].astype(np.float32)).reshape(1, 3)
K = torch.from_numpy(obj_info["K"].astype(np.float32))
# !!! pytorch3D is row-major
R = R.permute(0, 2, 1)
R[:, :, :2] *= -1
T[:, :2] *= -1


obj_mesh = trimesh.load("./sample_data/box.ply", force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)
# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * 0.5 / 2
verts = torch.from_numpy(obj_verts_can)
faces = torch.from_numpy(obj_mesh.faces)
mesh = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex([torch.ones_like(verts)])).to(device)


image_size = torch.tensor([480, 640]).unsqueeze(0)
cameras = PerspectiveCameras(focal_length=K[0, 0].reshape(-1, 1), principal_point=K[None, :2, -1], in_ndc=False, R=R, T=T, device=device, image_size=image_size)
raster_settings = RasterizationSettings(
    image_size=(480, 640), 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)


# render include the rasterizer and shader
# 光栅化的核心是cameras和相关设置
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
    )
)

images = renderer(mesh)
plt.figure()
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.savefig("./sample_data/render_res.jpg")