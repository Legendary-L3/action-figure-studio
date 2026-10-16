import base64
import os
import time
import tempfile
import trimesh
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
import requests

st.set_page_config(page_title="Action Figure 3D Studio", page_icon="🦸", layout="centered")

st.title("🦸 Action Figure 3D Studio")
st.caption("iPhone-to-Library Edition • Tripo3D Cloud AI & Live Joint Builder")

LIBRARY_SCALES = {
    "3.75-inch Vintage Scale (95mm) | ~1.5 - 2.5 hrs": 95.25,
    "5-inch Classic Scale (127mm) | ~3 - 4.5 hrs": 127.0,
    "6-inch Collector Scale (152mm) | ~5 - 7 hrs": 152.4,
    "7-inch Deluxe Scale (178mm) | ~6.5 - 8 hrs": 177.8,
}

JOINT_MODES = [
    "Poseable Ball-and-Socket Kit (Neck, Shoulders & Waist)",
    "Classic 5-Point Swivel Pegs (Retro Action Figure Style)",
    "Solid Statue (Single Uncut Piece)"
]

if "raw_mesh" not in st.session_state:
    st.session_state.raw_mesh = None

def render_3d_preview_on_iphone(glb_bytes: bytes):
    b64_glb = base64.b64encode(glb_bytes).decode("utf-8")
    html_code = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{b64_glb}"
        alt="3D Action Figure Preview"
        camera-controls
        shadow-intensity="1"
        style="width: 100%; height: 390px; background-color: #1e1e24; border-radius: 12px;">
    </model-viewer>
    """
    components.html(html_code, height=405)

def draw_2d_cut_overlay(img: Image.Image, neck_pct: float, waist_pct: float, shoulder_pct: float) -> Image.Image:
    preview = img.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    w, h = preview.size
    line_w = max(int(min(w, h) * 0.012), 3)

    neck_y = int(h * (1.0 - neck_pct))
    waist_y = int(h * (1.0 - waist_pct))
    left_x = int(w * (0.5 - shoulder_pct))
    right_x = int(w * (0.5 + shoulder_pct))

    draw.line([(0, neck_y), (w, neck_y)], fill=(255, 65, 65), width=line_w)
    draw.line([(0, waist_y), (w, waist_y)], fill=(45, 220, 90), width=line_w)
    draw.line([(left_x, neck_y), (left_x, waist_y)], fill=(65, 160, 255), width=line_w)
    draw.line([(right_x, neck_y), (right_x, waist_y)], fill=(65, 160, 255), width=line_w)
    return preview

def build_3d_cut_guide_preview(mesh: trimesh.Trimesh, neck_pct: float, waist_pct: float, shoulder_pct: float) -> bytes:
    scene = trimesh.Scene()
    body = mesh.copy()
    body.visual.face_colors = [215, 215, 225, 255]
    scene.add_geometry(body, node_name="figure")

    bounds = body.bounds
    min_pt, max_pt = bounds[0], bounds[1]
    width = max_pt[0] - min_pt[0]
    depth = max_pt[1] - min_pt[1]
    height = max_pt[2] - min_pt[2]
    cx = (min_pt[0] + max_pt[0]) / 2.0
    cy = (min_pt[1] + max_pt[1]) / 2.0

    neck_z = min_pt[2] + height * neck_pct
    waist_z = min_pt[2] + height * waist_pct
    shoulder_offset = width * shoulder_pct
    shoulder_z = min_pt[2] + height * ((neck_pct + waist_pct) / 2.05)

    neck_disc = trimesh.creation.cylinder(radius=max(width, depth) * 0.35, height=1.2, sections=48)
    neck_disc.apply_translation([cx, cy, neck_z])
    neck_disc.visual.face_colors = [255, 65, 65, 145]
    scene.add_geometry(neck_disc, node_name="neck_cut")

    waist_disc = trimesh.creation.cylinder(radius=max(width, depth) * 0.42, height=1.2, sections=48)
    waist_disc.apply_translation([cx, cy, waist_z])
    waist_disc.visual.face_colors = [45, 220, 90, 145]
    scene.add_geometry(waist_disc, node_name="waist_cut")

    sh_radius = height * 0.16
    rot_y = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    for idx, x_pos in enumerate([cx - shoulder_offset, cx + shoulder_offset]):
        sh_disc = trimesh.creation.cylinder(radius=sh_radius, height=1.2, sections=48)
        sh_disc.apply_transform(rot_y)
        sh_disc.apply_translation([x_pos, cy, shoulder_z])
        sh_disc.visual.face_colors = [65, 160, 255, 145]
        scene.add_geometry(sh_disc, node_name=f"shoulder_cut_{idx}")

    return scene.export(file_type="glb")

def generate_mesh_tripo(image_path: str) -> str:
    """Calls Tripo3D Cloud API with safe error handling."""
    api_key = st.secrets.get("TRIPO_API_KEY", "")
    if not api_key:
        raise ValueError("❌ TRIPO_API_KEY is missing! Add it in Streamlit Cloud Dashboard -> Settings -> Secrets.")

    headers = {"Authorization": f"Bearer {api_key}"}
    
    # 1. Upload image token
    with open(image_path, "rb") as f:
        upload_res = requests.post(
            "https://api.tripo3d.ai/v2/openapi/upload",
            headers=headers,
            files={"file": f}
        ).json()
    
    if "data" not in upload_res or "image_token" not in upload_res["data"]:
        raise RuntimeError(f"Tripo Upload Error Response: {upload_res}")
        
    image_token = upload_res["data"]["image_token"]

    # 2. Trigger task
    task_res = requests.post(
        "https://api.tripo3d.ai/v2/openapi/task",
        headers={**headers, "Content-Type": "application/json"},
        json={"type": "image_to_model", "file": {"type": "jpg", "file_token": image_token}}
    ).json()
    
    if "data" not in task_res or "task_id" not in task_res["data"]:
        raise RuntimeError(f"Tripo Task Creation Error Response: {task_res}")
        
    task_id = task_res["data"]["task_id"]

    # 3. Poll until complete (~30 sec)
    while True:
        status_res = requests.get(
            f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
            headers=headers
        ).json()
        
        if "data" not in status_res:
            raise RuntimeError(f"Tripo Polling Error Response: {status_res}")
            
        status = status_res["data"]["status"]
        if status == "success":
            glb_url = status_res["data"]["output"]["model"]
            break
        elif status in ["failed", "cancelled"]:
            raise RuntimeError(f"Tripo generation failed: {status_res}")
        time.sleep(3)

    glb_data = requests.get(glb_url).content
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    tmp.write(glb_data)
    tmp.close()
    return tmp.name

def make_joint_geometry(center, axis_vec, radius, is_ball=True, clearance=0.0):
    r = radius - clearance
    stem_len = radius * 1.6
    stem = trimesh.creation.cylinder(radius=r * 0.68, height=stem_len, sections=32)
    rot = trimesh.geometry.align_vectors([0, 0, 1], axis_vec)
    stem.apply_transform(rot)
    stem.apply_translation(center + np.array(axis_vec) * (stem_len * 0.35))
    if is_ball:
        ball = trimesh.creation.icosphere(subdivisions=3, radius=r)
        ball.apply_translation(center + np.array(axis_vec) * (stem_len * 0.75))
        return trimesh.util.concatenate([stem, ball])
    return stem

def cut_and_joint_pair(mesh_a, mesh_b, joint_center, normal_into_b, joint_radius, is_ball, clearance_mm):
    peg = make_joint_geometry(joint_center, normal_into_b, joint_radius, is_ball=is_ball, clearance=0.0)
    socket_cutter = make_joint_geometry(joint_center, normal_into_b, joint_radius, is_ball=is_ball, clearance=-clearance_mm)
    part_with_peg = trimesh.util.concatenate([mesh_a, peg])
    try:
        part_with_socket = trimesh.boolean.difference([mesh_b, socket_cutter], engine="manifold")
        if part_with_socket is None or len(part_with_socket.faces) == 0:
            part_with_socket = mesh_b
    except Exception:
        part_with_socket = mesh_b
    return part_with_peg, part_with_socket

def articulate_figure(mesh: trimesh.Trimesh, neck_pct: float, waist_pct: float, shoulder_pct: float, is_ball_joint: bool, clearance_mm: float = 0.25) -> trimesh.Trimesh:
    bounds = mesh.bounds
    min_pt, max_pt = bounds[0], bounds[1]
    width = max_pt[0] - min_pt[0]
    height = max_pt[2] - min_pt[2]
    center_x = (min_pt[0] + max_pt[0]) / 2.0
    center_y = (min_pt[1] + max_pt[1]) / 2.0

    neck_z = min_pt[2] + height * neck_pct
    waist_z = min_pt[2] + height * waist_pct
    shoulder_offset_x = width * shoulder_pct

    neck_r = max(height * 0.028, 2.8)
    waist_r = max(height * 0.036, 3.5)
    shoulder_r = max(height * 0.026, 2.6)

    head = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, 1], [center_x, center_y, neck_z], cap=True)
    below_neck = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, -1], [center_x, center_y, neck_z], cap=True)
    upper_body = trimesh.intersections.slice_mesh_plane(below_neck, [0, 0, 1], [center_x, center_y, waist_z], cap=True)
    legs = trimesh.intersections.slice_mesh_plane(below_neck, [0, 0, -1], [center_x, center_y, waist_z], cap=True)

    left_x = center_x - shoulder_offset_x
    right_x = center_x + shoulder_offset_x

    arm_left = trimesh.intersections.slice_mesh_plane(upper_body, [-1, 0, 0], [left_x, center_y, 0], cap=True)
    mid_torso = trimesh.intersections.slice_mesh_plane(upper_body, [1, 0, 0], [left_x, center_y, 0], cap=True)
    arm_right = trimesh.intersections.slice_mesh_plane(mid_torso, [1, 0, 0], [right_x, center_y, 0], cap=True)
    torso = trimesh.intersections.slice_mesh_plane(mid_torso, [-1, 0, 0], [right_x, center_y, 0], cap=True)

    shoulder_z = min_pt[2] + height * ((neck_pct + waist_pct) / 2.05)
    head, torso = cut_and_joint_pair(head, torso, [center_x, center_y, neck_z], [0, 0, -1], neck_r, is_ball_joint, clearance_mm)
    legs, torso = cut_and_joint_pair(legs, torso, [center_x, center_y, waist_z], [0, 0, 1], waist_r, is_ball_joint, clearance_mm)
    arm_left, torso = cut_and_joint_pair(arm_left, torso, [left_x, center_y, shoulder_z], [1, 0, 0], shoulder_r, is_ball_joint, clearance_mm)
    arm_right, torso = cut_and_joint_pair(arm_right, torso, [right_x, center_y, shoulder_z], [-1, 0, 0], shoulder_r, is_ball_joint, clearance_mm)

    gap = height * 0.08
    head.apply_translation([0, 0, gap * 1.2])
    legs.apply_translation([0, 0, -gap * 0.5])
    arm_left.apply_translation([-gap * 1.1, 0, 0])
    arm_right.apply_translation([gap * 1.1, 0, 0])

    assembled_kit = trimesh.util.concatenate([head, torso, arm_left, arm_right, legs])
    min_z = assembled_kit.bounds[0][2]
    assembled_kit.apply_translation([0, 0, -min_z])
    return assembled_kit

# --- UI ---
st.subheader("Step 1: Upload Photo & Generate Base 3D Mesh")
uploaded_file = st.file_uploader("Tap to Snap Photo or Pick from iPhone Camera Roll", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    if st.button("✨ Step 1: Generate Character 3D Model", type="primary", use_container_width=True):
        with st.spinner("Tripo3D AI is sculpting your 2D photo into 3D (~30 sec)..."):
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(temp_img.name, "PNG")
            try:
                glb_path = generate_mesh_tripo(temp_img.name)
                mesh = trimesh.load(glb_path, force="mesh")
                mesh.fill_holes()
                mesh.fix_normals()
                min_z = mesh.bounds[0][2]
                mesh.apply_translation([0, 0, -min_z])
                mesh.apply_scale(152.4 / mesh.extents[2])
                st.session_state.raw_mesh = mesh
                st.success("3D Character Model Generated Successfully!")
            except Exception as e:
                st.error(f"{e}")

    if st.session_state.raw_mesh is not None:
        st.divider()
        st.subheader("Step 2: Fine-Tune Joint Cut Lines")
        st.caption("🔴 Red = Neck Cut • 🟢 Green = Waist Cut • 🔵 Blue = Shoulder Cuts")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            neck_pct = st.slider("🔴 Neck Height %", 70, 90, 81, 1) / 100.0
        with col_s2:
            waist_pct = st.slider("🟢 Waist Height %", 35, 62, 47, 1) / 100.0
        with col_s3:
            shoulder_pct = st.slider("🔵 Shoulder Offset %", 14, 38, 23, 1) / 100.0

        overlay_2d = draw_2d_cut_overlay(img, neck_pct, waist_pct, shoulder_pct)
        st.image(overlay_2d, caption="2D Cut Plane Alignment Preview", width=260)

        st.markdown("**🔄 Live 3D Cut-Ring Preview:**")
        guide_glb = build_3d_cut_guide_preview(st.session_state.raw_mesh, neck_pct, waist_pct, shoulder_pct)
        render_3d_preview_on_iphone(guide_glb)

        st.subheader("Step 3: Scale, Carve Joints & Download .STL")
        scale_label = st.selectbox("Select Final Print Scale", options=list(LIBRARY_SCALES.keys()), index=2)
        joint_mode = st.radio("Select Joint Style", options=JOINT_MODES, index=0)

        c1, c2 = st.columns(2)
        with c1:
            add_base = st.checkbox("Weld Flat Stand Base", value=True)
        with c2:
            clearance = st.slider("Joint Clearance (mm)", 0.15, 0.40, 0.25, 0.05)

        if st.button("🔪 Slice Joints & Export Library .STL", type="primary", use_container_width=True):
            with st.spinner("Carving boolean sockets & exporting STL..."):
                final_mesh = st.session_state.raw_mesh.copy()
                target_mm = LIBRARY_SCALES[scale_label]
                final_mesh.apply_scale(target_mm / final_mesh.extents[2])

                if "Ball-and-Socket" in joint_mode:
                    final_mesh = articulate_figure(final_mesh, neck_pct, waist_pct, shoulder_pct, is_ball_joint=True, clearance_mm=clearance)
                elif "Swivel Pegs" in joint_mode:
                    final_mesh = articulate_figure(final_mesh, neck_pct, waist_pct, shoulder_pct, is_ball_joint=False, clearance_mm=clearance)

                if add_base:
                    radius = max(final_mesh.extents[0], final_mesh.extents[1]) * 0.42
                    base_h = 3.2
                    base = trimesh.creation.cylinder(radius=radius, height=base_h, sections=72)
                    base.apply_translation([0, 0, base_h / 2.0])
                    final_mesh.apply_translation([0, 0, base_h - 0.8])
                    final_mesh = trimesh.util.concatenate([final_mesh, base])

                volume_cm3 = abs(final_mesh.volume) / 1000.0
                est_grams = round(max(volume_cm3 * 1.24 * 0.28, 15.0), 1)

                final_glb = final_mesh.export(file_type="glb")
                final_stl = final_mesh.export(file_type="stl")
                clean_tag = scale_label.split("-inch")[0].replace(".", "_") + "in"
                file_name = f"poseable_figure_{clean_tag}.stl"

            st.success("Joints Carved Successfully!")
            st.markdown("**🔄 Final Exploded Parts Kit:**")
            render_3d_preview_on_iphone(final_glb)

            st.info(
                f"**Library Slicer Checklist:**\n"
                f"- **Est. PLA Filament:** ~{est_grams} grams\n"
                f"- **Maker Lab Tip:** Right-click model in slicer -> **'Split to Objects'** -> **'Auto-Orient'**."
            )

            st.download_button(
                label=f"📥 Save {file_name} to iPhone Files",
                data=final_stl,
                file_name=file_name,
                mime="application/sla",
                type="primary",
                use_container_width=True
            )
