import os
import numpy as np
import torch
import trimesh
import cv2
import gradio as gr
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams
)
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures.utils import padded_to_packed
from pytorch3d.transforms import axis_angle_to_matrix
import torch.nn as nn
from typing import Optional
from pytorch3d.renderer.camera_utils import rotate_on_spot
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Shader class ---
class NormalShader(nn.Module):
    def __init__(self, blend_params: Optional[BlendParams] = None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, cameras=None):
        if cameras is None:
            verts_normal = meshes.verts_normals_padded()
        else:
            world_to_view_transform = cameras.get_world_to_view_transform()
            verts_normal = world_to_view_transform.transform_normals(
                meshes.verts_normals_padded())
        faces = meshes.faces_packed()
        verts_normal = padded_to_packed(verts_normal)
        faces_normal = verts_normal[faces]
        normal_map = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_normal)
        return normal_map[..., 0, :]

# --- Ambient light function ---
def ambient_light(device='cpu', cameras=None, light_color=np.array([0.65, 0.3, 0.0])):
    d = torch.FloatTensor([[0, 1, -1]]).to(device)
    N = 1 if cameras is None else len(cameras)
    zeros = torch.zeros([N, 3], device=device)
    d = zeros + d
    if cameras is not None:
        d = cameras.get_world_to_view_transform().inverse().transform_normals(d.unsqueeze(1)).squeeze(1)
    am, df, sp = light_color
    am = zeros + am
    df = zeros + df
    sp = zeros + sp
    lights = DirectionalLights(
        device=device,
        ambient_color=am,
        diffuse_color=df,
        specular_color=sp,
        direction=d
    )
    return lights

# --- Render function ---
def render_mesh(mesh_path, distance, elev, azim, roll_rotation, save_path):
    mesh_obj = trimesh.load(mesh_path, force="mesh")
    verts = np.array(mesh_obj.vertices).astype(np.float32)
    verts = verts - verts.mean(0)
    verts = verts / np.linalg.norm(verts, 2, 1).max() * 0.5 / 2
    verts = torch.from_numpy(verts).to(device)
    faces = torch.from_numpy(np.array(mesh_obj.faces)).to(device)
    mesh = Meshes(verts=[verts], faces=[faces],
                  textures=TexturesVertex([torch.ones_like(verts)])).to(device)
    
    # Set vertex color
    color = torch.FloatTensor([[[235/255, 245/255, 246/255]]]).to(mesh.device) * 2 - 1
    feature = torch.zeros_like(mesh.verts_padded()) + color
    texture = TexturesVertex(feature)
    texture._num_faces_per_mesh = mesh.num_faces_per_mesh().tolist()
    texture._num_verts_per_mesh = mesh.num_verts_per_mesh().tolist()
    texture._N = mesh._N
    texture.valid = mesh.valid
    mesh.textures = texture

    # Camera
    R, T = look_at_view_transform(distance, elev, azim)
    roll_rotation = axis_angle_to_matrix(torch.FloatTensor([0, 0, math.radians(roll_rotation)])).to(R.device)
    R, T = rotate_on_spot(R, T, roll_rotation)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Rasterizer & shader
    raster_settings = RasterizationSettings( image_size=(1024, 1024), blur_radius=0.0, faces_per_pixel=1, )
    rasterizer=MeshRasterizer(
        raster_settings=raster_settings
    )
    materials = Materials(
        specular_color=((0.2, 0.2, 0.2),), # 默认是1,1,1，就是纯白色；测试发现调成0.2,0.2,0.2比较适合人的皮肤。
        shininess=30, # 默认值是 64，看上去高光稍微有点聚集了，改成30的话略自然，差别不太明显
        device=device,
    )
    shader = SoftPhongShader(device, materials=materials)
    
    fragments = rasterizer(meshes_world=mesh, cameras=cameras)
    image = shader(fragments, mesh, cameras=cameras, lights=ambient_light(mesh.device, cameras))
    image = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    
    cv2.imwrite(save_path, image)
    return image[:, :, ::-1]



if __name__ == '__main__':
    import os
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, required=True)
    args = parser.parse_args()
    def render_mesh_wrapper(distance, elev, azim, roll_rotation, save_path):
        return render_mesh(args.obj_path, distance, elev, azim, roll_rotation, save_path)
    # --- Gradio UI ---
    ui = gr.Interface(
        fn=render_mesh_wrapper,
        inputs=[
            gr.Slider(0.1, 5.0, value=0.8, step=0.01, label="Distance"),
            gr.Slider(-180, 180, value=0, step=1, label="Elevation"),
            gr.Slider(-180, 180, value=40, step=1, label="Azimuth"),
            gr.Slider(-180, 180, value=0, step=1, label="Roll"),
            gr.Textbox(value="render_res.png", label="Save Path")
        ],
        outputs=gr.Image(type="numpy"),
        live=True,
        title="PyTorch3D Mesh Renderer"
    )


    ui.launch(
        server_name="0.0.0.0",   # 外部可访问
        server_port=7860,
        share=False,             # 禁用公网
        inbrowser=False,         # 不尝试在服务器端打开浏览器
        prevent_thread_lock=True # 防止阻塞
    )


