import os
import json
import numpy as np
import torch
import trimesh
import cv2
# Compatibility guard for some gradio/pydantic setups which directly index this env var.
os.environ.setdefault("PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES", "1")
import gradio as gr
from pytorch3d.structures import Meshes, Pointclouds
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
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
    BlendParams
)
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures.utils import padded_to_packed
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
import torch.nn as nn
from typing import Optional
import math


device = "cuda" if torch.cuda.is_available() else "cpu"
MIN_DISTANCE = 0.2
MAX_DISTANCE = 5.0
FRONTEND_CSS = """
.arcball-hidden {
  position: absolute !important;
  left: -10000px !important;
  top: -10000px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
#render_view {
  position: relative !important;
}
#render_view .arcball-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: auto;
  cursor: grab;
  touch-action: none;
  z-index: 20;
}
"""

FRONTEND_SCRIPT = """
<script>
(() => {
  if (window.__uiRenderArcballInitialized) return;
  window.__uiRenderArcballInitialized = true;

  const MIN_DISTANCE = 0.2;
  const MAX_DISTANCE = 5.0;
  const THROTTLE_MS = 50;
  let currentState = null;
  let dragging = false;
  let lastVec = null;
  let pendingTimer = null;
  let latestPayload = null;
  let windowPointerListenersBound = false;
  let overlayCanvas = null;

  function getInputByElemId(elemId) {
    const root = document.getElementById(elemId);
    if (!root) return null;
    return root.querySelector("textarea, input");
  }

  function normalizeQuat(q) {
    const n = Math.hypot(q[0], q[1], q[2], q[3]);
    if (n < 1e-8) return [1, 0, 0, 0];
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  }

  function quatMul(a, b) {
    const [aw, ax, ay, az] = a;
    const [bw, bx, by, bz] = b;
    return [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ];
  }

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function normalizeVec(v) {
    const n = Math.hypot(v[0], v[1], v[2]);
    if (n < 1e-8) return [0, 0, 1];
    return [v[0] / n, v[1] / n, v[2] / n];
  }

  function quatConj(q) {
    return [q[0], -q[1], -q[2], -q[3]];
  }

  function rotateVecByQuat(v, q) {
    const qv = [0.0, v[0], v[1], v[2]];
    const t = quatMul(quatMul(q, qv), quatConj(q));
    return [t[1], t[2], t[3]];
  }

  function quatFromVectors(a, b) {
    const va = normalizeVec(a);
    const vb = normalizeVec(b);
    const c = cross(va, vb);
    let w = dot(va, vb) + 1.0;
    if (w < 1e-6) {
      let axis;
      if (Math.abs(va[0]) > Math.abs(va[2])) {
        axis = normalizeVec([-va[1], va[0], 0.0]);
      } else {
        axis = normalizeVec([0.0, -va[2], va[1]]);
      }
      return [0.0, axis[0], axis[1], axis[2]];
    }
    return normalizeQuat([w, c[0], c[1], c[2]]);
  }

  function projectToSphere(clientX, clientY, surface) {
    const rect = surface.getBoundingClientRect();
    const x = ((clientX - rect.left) / Math.max(rect.width, 1)) * 2.0 - 1.0;
    const y = 1.0 - ((clientY - rect.top) / Math.max(rect.height, 1)) * 2.0;
    const d2 = x * x + y * y;
    if (d2 <= 1.0) {
      return [x, y, Math.sqrt(1.0 - d2)];
    }
    const invD = 1.0 / Math.sqrt(d2);
    return [x * invD, y * invD, 0.0];
  }

  function parseStateFromPython() {
    const input = getInputByElemId("camera_state_json");
    if (!input || !input.value) return null;
    try {
      const parsed = JSON.parse(input.value);
      if (!Array.isArray(parsed.arcball_quat) || parsed.arcball_quat.length !== 4) {
        parsed.arcball_quat = [1, 0, 0, 0];
      }
      parsed.arcball_quat = normalizeQuat(parsed.arcball_quat);
      parsed.distance = Math.min(MAX_DISTANCE, Math.max(MIN_DISTANCE, Number(parsed.distance ?? 0.8)));
      return parsed;
    } catch (_err) {
      return null;
    }
  }

  function emitPayload(payload) {
    const input = getInputByElemId("js_camera_event");
    if (!input) return;
    input.value = JSON.stringify(payload);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function emitThrottled(payload) {
    latestPayload = payload;
    if (pendingTimer !== null) return;
    pendingTimer = window.setTimeout(() => {
      pendingTimer = null;
      if (latestPayload) emitPayload(latestPayload);
      latestPayload = null;
    }, THROTTLE_MS);
  }

  function syncState() {
    const s = parseStateFromPython();
    if (s) currentState = s;
    if (!currentState) {
      currentState = { distance: 0.8, arcball_quat: [1, 0, 0, 0], elev: 0, azim: 0, roll: 0 };
    }
    drawTrackball();
  }

  function resizeCanvasToImage(viewportEl, canvas) {
    const rect = viewportEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(2, Math.round(rect.width));
    const h = Math.max(2, Math.round(rect.height));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function getViewportElement() {
    const renderRoot = document.getElementById("render_view");
    if (!renderRoot) return null;
    return renderRoot.querySelector("img, canvas:not(.arcball-overlay)");
  }

  function ensureOverlayCanvas(viewportEl) {
    const renderRoot = document.getElementById("render_view");
    if (!renderRoot || !viewportEl) return null;
    if (!overlayCanvas) {
      overlayCanvas = document.createElement("canvas");
      overlayCanvas.className = "arcball-overlay";
      renderRoot.appendChild(overlayCanvas);
    }
    resizeCanvasToImage(viewportEl, overlayCanvas);
    return overlayCanvas;
  }

  function drawRing(ctx, q, centerX, centerY, radius, baseAxis, color) {
    const samples = 180;
    ctx.beginPath();
    for (let i = 0; i <= samples; i++) {
      const t = (Math.PI * 2 * i) / samples;
      let p;
      if (baseAxis === "x") p = [0, Math.cos(t), Math.sin(t)];
      else if (baseAxis === "y") p = [Math.cos(t), 0, Math.sin(t)];
      else p = [Math.cos(t), Math.sin(t), 0];
      const pr = rotateVecByQuat(p, q);
      const x = centerX + pr[0] * radius;
      const y = centerY - pr[1] * radius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  function drawTrackball() {
    const viewportEl = getViewportElement();
    if (!viewportEl) return;
    const canvas = ensureOverlayCanvas(viewportEl);
    if (!canvas || !currentState) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = parseFloat(canvas.style.width) || viewportEl.getBoundingClientRect().width || 2;
    const h = parseFloat(canvas.style.height) || viewportEl.getBoundingClientRect().height || 2;
    const cx = w * 0.5;
    const cy = h * 0.5;
    const r = Math.max(20, Math.min(w, h) * 0.38);
    const q = normalizeQuat(currentState.arcball_quat || [1, 0, 0, 0]);

    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    drawRing(ctx, q, cx, cy, r, "x", "rgba(255, 110, 110, 0.85)");
    drawRing(ctx, q, cx, cy, r, "y", "rgba(120, 235, 140, 0.85)");
    drawRing(ctx, q, cx, cy, r, "z", "rgba(120, 170, 255, 0.85)");
  }

  function onPointerDown(ev, surface) {
    if (ev.button !== 0) return;
    syncState();
    dragging = true;
    lastVec = projectToSphere(ev.clientX, ev.clientY, surface);
    surface.style.cursor = "grabbing";
    ev.preventDefault();
  }

  function onPointerMove(ev, surface) {
    if (!dragging || !lastVec) return;
    const curVec = projectToSphere(ev.clientX, ev.clientY, surface);
    const qDelta = quatFromVectors(lastVec, curVec);
    const nextQuat = normalizeQuat(quatMul(qDelta, currentState.arcball_quat || [1, 0, 0, 0]));
    currentState.arcball_quat = nextQuat;
    drawTrackball();
    lastVec = curVec;
    emitThrottled(currentState);
    ev.preventDefault();
  }

  function onWheel(ev) {
    syncState();
    const factor = Math.exp(ev.deltaY * 0.001);
    currentState.distance = Math.min(MAX_DISTANCE, Math.max(MIN_DISTANCE, currentState.distance * factor));
    emitThrottled(currentState);
    ev.preventDefault();
  }

  function bindArcball(viewportEl) {
    const surface = ensureOverlayCanvas(viewportEl);
    if (!surface || surface.dataset.arcballBound === "1") return;
    surface.dataset.arcballBound = "1";
    surface.style.cursor = "grab";
    surface.addEventListener("pointerdown", (ev) => onPointerDown(ev, surface), { passive: false });
    surface.addEventListener("pointermove", (ev) => onPointerMove(ev, surface), { passive: false });
    surface.addEventListener("wheel", onWheel, { passive: false });
    viewportEl.addEventListener("load", () => drawTrackball());
    window.addEventListener("resize", () => drawTrackball());
    drawTrackball();
    if (!windowPointerListenersBound) {
      windowPointerListenersBound = true;
      window.addEventListener("pointerup", () => {
        dragging = false;
        lastVec = null;
        if (overlayCanvas) overlayCanvas.style.cursor = "grab";
      });
      window.addEventListener("pointercancel", () => {
        dragging = false;
        lastVec = null;
        if (overlayCanvas) overlayCanvas.style.cursor = "grab";
      });
    }
  }

  function tryAttach() {
    syncState();
    const renderRoot = document.getElementById("render_view");
    if (!renderRoot) {
      window.setTimeout(tryAttach, 300);
      return;
    }
    const bindLatestImage = () => {
      const viewportEl = getViewportElement();
      if (viewportEl) bindArcball(viewportEl);
    };
    bindLatestImage();
    const observer = new MutationObserver(() => {
      syncState();
      bindLatestImage();
    });
    observer.observe(renderRoot, { childList: true, subtree: true });

    const stateInput = getInputByElemId("camera_state_json");
    if (stateInput) {
      stateInput.addEventListener("input", syncState);
      stateInput.addEventListener("change", syncState);
    }
  }

  window.setTimeout(tryAttach, 300);
})();
</script>
"""

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

def default_camera_state():
    return {
        "distance": 0.8,
        "arcball_quat": [1.0, 0.0, 0.0, 0.0],
        "elev": 0.0,
        "azim": 0.0,
        "roll": 0.0,
    }

def clamp_distance(distance):
    return float(max(MIN_DISTANCE, min(MAX_DISTANCE, float(distance))))

def normalize_quaternion(quat):
    quat = torch.as_tensor(quat, dtype=torch.float32, device=device)
    if quat.numel() != 4:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    norm = torch.linalg.norm(quat)
    if norm < 1e-8:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    else:
        quat = quat / norm
    return quat

def ensure_camera_state(camera_state):
    state = dict(default_camera_state())
    if isinstance(camera_state, dict):
        state.update(camera_state)
    state["distance"] = clamp_distance(state.get("distance", 0.8))
    state["elev"] = float(state.get("elev", 0.0))
    state["azim"] = float(state.get("azim", 0.0))
    state["roll"] = float(state.get("roll", 0.0))
    quat = normalize_quaternion(state.get("arcball_quat", [1, 0, 0, 0]))
    state["arcball_quat"] = quat.detach().cpu().tolist()
    return state

def _wrap_deg(value):
    return ((float(value) + 180.0) % 360.0) - 180.0

def quaternion_from_euler_deg(elev, azim, roll):
    angles = torch.tensor(
        [math.radians(float(elev)), math.radians(float(azim)), math.radians(float(roll))],
        dtype=torch.float32,
        device=device,
    )
    rot = euler_angles_to_matrix(angles, "XYZ")
    quat = matrix_to_quaternion(rot)
    quat = quat / (torch.linalg.norm(quat) + 1e-8)
    return quat

def euler_deg_from_quaternion(quat):
    quat = normalize_quaternion(quat)
    rot = quaternion_to_matrix(quat.unsqueeze(0))[0]
    angles = matrix_to_euler_angles(rot, "XYZ")
    elev, azim, roll = [math.degrees(v.item()) for v in angles]
    return _wrap_deg(elev), _wrap_deg(azim), _wrap_deg(roll)

def sync_quaternion_from_euler(state):
    quat = quaternion_from_euler_deg(state["elev"], state["azim"], state["roll"])
    state["arcball_quat"] = quat.detach().cpu().tolist()
    return state

def sync_euler_from_quaternion(state):
    elev, azim, roll = euler_deg_from_quaternion(state["arcball_quat"])
    state["elev"] = elev
    state["azim"] = azim
    state["roll"] = roll
    return state

def apply_arcball_rotation(verts, camera_state):
    quat = normalize_quaternion(camera_state["arcball_quat"])
    arcball_rot = quaternion_to_matrix(quat.unsqueeze(0))[0]
    return torch.matmul(verts, arcball_rot.T)

def build_cameras(camera_state):
    distance = camera_state["distance"]
    R, T = look_at_view_transform(distance, 0.0, 0.0)
    return FoVPerspectiveCameras(device=device, R=R, T=T)

def render_mesh(mesh_path, camera_state):
    camera_state = ensure_camera_state(camera_state)
    mesh_obj = trimesh.load(mesh_path, force="mesh")
    verts = np.array(mesh_obj.vertices).astype(np.float32)
    verts = verts - verts.mean(0)
    verts = verts / np.linalg.norm(verts, 2, 1).max() * 0.5 / 2
    verts = torch.from_numpy(verts).to(device)
    verts = apply_arcball_rotation(verts, camera_state)
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
    cameras = build_cameras(camera_state)

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
    return image[:, :, ::-1]

def render_pc(pc_path, camera_state):
    camera_state = ensure_camera_state(camera_state)
    # 加载点云
    mesh_obj = trimesh.load(pc_path)
    verts = np.array(mesh_obj.vertices).astype(np.float32)

    # 中心化 + 归一化
    verts = verts - verts.mean(0)
    verts = verts / np.linalg.norm(verts, 2, 1).max() * 0.25

    verts = torch.from_numpy(verts).to(device)
    verts = apply_arcball_rotation(verts, camera_state)

    # 点颜色（统一颜色，可换成 per-vertex color）
    color = torch.tensor([[235/255, 245/255, 246/255]], device=device).repeat(verts.shape[0], 1)
    point_cloud = Pointclouds(points=[verts], features=[color])

    # 相机参数
    cameras = build_cameras(camera_state)

    # 光栅化设置
    raster_settings = PointsRasterizationSettings(
        image_size=1024,
        radius=0.003,     # 点的半径，调节大小
        points_per_pixel=10
    )

    # 渲染器
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    # 渲染
    images = renderer(point_cloud)
    image = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]  # BGR for cv2

    return image[:, :, ::-1]   # 返回 RGB

def normalize_save_path(save_path):
    save_path = (save_path or "").strip()
    if not save_path:
        save_path = "render_res.png"
    save_path = os.path.expanduser(save_path)
    if not os.path.splitext(save_path)[1]:
        save_path += ".png"
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    return save_path

def save_current_image(image_np, save_path):
    if image_np is None:
        return "No image to save."
    try:
        final_path = normalize_save_path(save_path)
        bgr = image_np[:, :, ::-1]
        ok = cv2.imwrite(final_path, bgr)
        if not ok:
            return f"Save failed: cannot write to {final_path}"
        return f"Saved: {final_path}"
    except Exception as err:
        return f"Save failed: {err}"


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

    tmp_input = trimesh.load(args.obj_path)
    is_point_cloud = isinstance(tmp_input, trimesh.PointCloud) or (
        hasattr(tmp_input, "faces") and len(tmp_input.faces) == 0
    )

    def render_scene(camera_state):
        if is_point_cloud:
            return render_pc(args.obj_path, camera_state)
        else:
            return render_mesh(args.obj_path, camera_state)

    def render_from_controls(distance, elev, azim, roll, camera_state):
        state = ensure_camera_state(camera_state)
        state["distance"] = clamp_distance(distance)
        state["elev"] = float(elev)
        state["azim"] = float(azim)
        state["roll"] = float(roll)
        sync_quaternion_from_euler(state)
        try:
            image = render_scene(state)
            status = "Preview updated."
        except Exception as err:
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            status = f"Render failed: {err}"
        return image, state, state["distance"], state["elev"], state["azim"], state["roll"], json.dumps(state), status

    def render_from_js_event(js_payload, elev, azim, roll, camera_state):
        state = ensure_camera_state(camera_state)
        state["elev"] = float(elev)
        state["azim"] = float(azim)
        state["roll"] = float(roll)
        if js_payload:
            try:
                payload = json.loads(js_payload)
                if "distance" in payload:
                    state["distance"] = clamp_distance(payload["distance"])
                if "arcball_quat" in payload:
                    state["arcball_quat"] = normalize_quaternion(payload["arcball_quat"]).detach().cpu().tolist()
                    sync_euler_from_quaternion(state)
            except Exception:
                pass
        try:
            image = render_scene(state)
            status = "Preview updated from mouse interaction."
        except Exception as err:
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
            status = f"Render failed: {err}"
        return image, state, state["distance"], state["elev"], state["azim"], state["roll"], json.dumps(state), status

    init_state = default_camera_state()

    with gr.Blocks(title="PyTorch3D Mesh Renderer", head=FRONTEND_SCRIPT, css=FRONTEND_CSS) as ui:
        camera_state = gr.State(init_state)
        with gr.Row():
            render_view = gr.Image(type="numpy", label="Render Preview", elem_id="render_view")
            with gr.Column():
                gr.Markdown("Drag directly on the preview to rotate (arcball). Use mouse wheel to zoom.")
                with gr.Accordion("Advanced Numeric Controls", open=False):
                    distance = gr.Slider(
                        MIN_DISTANCE, MAX_DISTANCE, value=init_state["distance"], step=0.01, label="Distance"
                    )
                    elev = gr.Slider(-180, 180, value=init_state["elev"], step=1, label="Elevation")
                    azim = gr.Slider(-180, 180, value=init_state["azim"], step=1, label="Azimuth")
                    roll = gr.Slider(-180, 180, value=init_state["roll"], step=1, label="Roll")
                save_path = gr.Textbox(value="render_res.png", label="Save Path")
                save_btn = gr.Button("Save Current View")
                status_box = gr.Textbox(label="Status", interactive=False)

        camera_state_json = gr.Textbox(
            value=json.dumps(init_state), elem_id="camera_state_json", elem_classes=["arcball-hidden"]
        )
        js_camera_event = gr.Textbox(value="", elem_id="js_camera_event", elem_classes=["arcball-hidden"])

        ui.load(
            fn=render_from_controls,
            inputs=[distance, elev, azim, roll, camera_state],
            outputs=[render_view, camera_state, distance, elev, azim, roll, camera_state_json, status_box],
            queue=False
        )

        for comp in [distance, elev, azim, roll]:
            comp.change(
                fn=render_from_controls,
                inputs=[distance, elev, azim, roll, camera_state],
                outputs=[render_view, camera_state, distance, elev, azim, roll, camera_state_json, status_box],
                queue=False
            )

        js_camera_event.input(
            fn=render_from_js_event,
            inputs=[js_camera_event, elev, azim, roll, camera_state],
            outputs=[render_view, camera_state, distance, elev, azim, roll, camera_state_json, status_box],
            queue=False
        )
        js_camera_event.change(
            fn=render_from_js_event,
            inputs=[js_camera_event, elev, azim, roll, camera_state],
            outputs=[render_view, camera_state, distance, elev, azim, roll, camera_state_json, status_box],
            queue=False
        )

        save_btn.click(
            fn=save_current_image,
            inputs=[render_view, save_path],
            outputs=[status_box],
            queue=False
        )


    ui.launch(
        server_name="0.0.0.0",   # 外部可访问
        server_port=7860,
        share=False,             # 禁用公网
        inbrowser=False,         # 不尝试在服务器端打开浏览器
        prevent_thread_lock=False # 防止阻塞
    )
