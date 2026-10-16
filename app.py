import base64
import tempfile
import trimesh
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from gradio_client import Client, handle_file

st.set_page_config(page_title="Action Figure 3D Studio", page_icon="🦸", layout="centered")

st.title("🦸 Action Figure 3D Studio")
st.caption("iPhone-to-Library Edition • Poseable Joints & Statue Builder")

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

def render_3d_preview_on_iphone(glb_bytes: bytes):
    """Renders an interactive touch-enabled 3D turntable in iPhone Safari."""
    b64_glb = base64.b64encode(glb_bytes).decode("utf-8")
    html_code = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{b64_glb}"
        alt="3D Action Figure Preview"
        auto-rotate
        camera-controls
        shadow-intensity="1"
        style="width: 100%; height: 380px; background-color: #1e1e24; border-radius: 12px;">
    </model-viewer>
    """
    components.html(html_code, height=395)

def generate_mesh_hunyuan(image_path: str) -> str:
    """Calls Tencent Hunyuan3D-2 Space remotely for free 2D-to-3D generation."""
    hf_token = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None
    client = Client("tencent/Hunyuan3D-2", hf_token=hf_token)
    result = client.predict(
        image=handle_file(image_path),
        api_name="/shape_generation"
    )
    if isinstance(result, (list, tuple)):
        glb_file = result[0]
        return glb_file["value"] if isinstance(glb_file, dict) and "value" in glb_file else glb_file
    elif isinstance(result, dict) and "value" in result:
        return result["value"]
    return result

def build_fallback_mannequin() -> trimesh.Trimesh:
    """Fallback humanoid figure with arms in A-pose so you can test joint slicing anytime."""
    torso = trimesh.creation.capsule(height=0.55, radius=0.16)
    torso.apply_translation([0, 0, 0.78])
    head = trimesh.creation.icosphere(radius=0.14)
    head.apply_translation([0, 0, 1.28])
    
    # Left & Right Arms extended enough for clean shoulder cuts
    arm_l = trimesh.creation.capsule(height=0.45, radius=0.065)
    arm_l.apply_translation([-0.26, 0, 0.80])
    arm_r = trimesh.creation.capsule(height=0.45, radius=0.065)
    arm_r.apply_translation([0.26, 0, 0.80])
    
    leg_l = trimesh.creation.capsule(height=0.50, radius=0.075)
    leg_l.apply_translation([-0.09, 0, 0.28])
    leg_r = trimesh.creation.capsule(height=0.50, radius=0.075)
    leg_r.apply_translation([0.09, 0, 0.28])
    return trimesh.util.concatenate([torso, head, arm_l, arm_r, leg_l, leg_r])

def make_joint_geometry(center, axis_vec, radius, is_ball=True, clearance=0.0):
    """Creates a ball-stem peg or cylindrical swivel peg oriented along axis_vec."""
    r = radius - clearance
    stem_len = radius * 1.6
    stem = trimesh.creation.cylinder(radius=r * 0.68, height=stem_len, sections=32)
    
    # Align cylinder along axis_vec
    rot = trimesh.geometry.align_vectors([0, 0, 1], axis_vec)
    stem.apply_transform(rot)
    stem.apply_translation(center + np.array(axis_vec) * (stem_len * 0.35))
    
    if is_ball:
        ball = trimesh.creation.icosphere(subdivisions=3, radius=r)
        ball.apply_translation(center + np.array(axis_vec) * (stem_len * 0.75))
        return trimesh.util.concatenate([stem, ball])
    return stem

def cut_and_joint_pair(mesh_a, mesh_b, joint_center, normal_into_b, joint_radius, is_ball, clearance_mm):
    """
    Adds a protruding peg/ball onto mesh_a and carves a matching socket out of mesh_b.
    Falls back cleanly if boolean backend hits non-manifold AI geometry.
    """
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

def articulate_figure(mesh: trimesh.Trimesh, is_ball_joint: bool, clearance_mm: float = 0.25) -> trimesh.Trimesh:
    """
    Slices the scaled figure into Head, Torso, Left Arm, Right Arm, and Lower Body,
    adds snap-fit joints, and arranges them in an exploded layout for inspection & printing.
    """
    bounds = mesh.bounds
    min_pt, max_pt = bounds[0], bounds[1]
    width = max_pt[0] - min_pt[0]
    height = max_pt[2] - min_pt[2]
    center_x = (min_pt[0] + max_pt[0]) / 2.0
    center_y = (min_pt[1] + max_pt[1]) / 2.0

    # Proportional cut planes
    neck_z = min_pt[2] + height * 0.81
    waist_z = min_pt[2] + height * 0.47
    shoulder_offset_x = width * 0.23

    # Scale joint radius to figure size (e.g., ~4.5mm ball for 6-inch figure)
    neck_r = max(height * 0.028, 2.8)
    waist_r = max(height * 0.036, 3.5)
    shoulder_r = max(height * 0.026, 2.6)

    # 1. Slice Head vs Body
    head = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, 1], [center_x, center_y, neck_z], cap=True)
    below_neck = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, -1], [center_x, center_y, neck_z], cap=True)

    # 2. Slice Upper Body vs Lower Body (Waist)
    upper_body = trimesh.intersections.slice_mesh_plane(below_neck, [0, 0, 1], [center_x, center_y, waist_z], cap=True)
    legs = trimesh.intersections.slice_mesh_plane(below_neck, [0, 0, -1], [center_x, center_y, waist_z], cap=True)

    # 3. Slice Left & Right Arms from Torso
    left_x = center_x - shoulder_offset_x
    right_x = center_x + shoulder_offset_x

    arm_left = trimesh.intersections.slice_mesh_plane(upper_body, [-1, 0, 0], [left_x, center_y, 0], cap=True)
    mid_torso = trimesh.intersections.slice_mesh_plane(upper_body, [1, 0, 0], [left_x, center_y, 0], cap=True)
    arm_right = trimesh.intersections.slice_mesh_plane(mid_torso, [1, 0, 0], [right_x, center_y, 0], cap=True)
    torso = trimesh.intersections.slice_mesh_plane(mid_torso, [-1, 0, 0], [right_x, center_y, 0], cap=True)

    # 4. Engineer Ball-and-Socket or Swivel Pegs at each cut
    shoulder_z = min_pt[2] + height * 0.68
    head, torso = cut_and_joint_pair(
        head, torso, [center_x, center_y, neck_z], [0, 0, -1], neck_r, is_ball_joint, clearance_mm
    )
    legs, torso = cut_and_joint_pair(
        legs, torso, [center_x, center_y, waist_z], [0, 0, 1], waist_r, is_ball_joint, clearance_mm
    )
    arm_left, torso = cut_and_joint_pair(
        arm_left, torso, [left_x, center_y, shoulder_z], [1, 0, 0], shoulder_r, is_ball_joint, clearance_mm
    )
    arm_right, torso = cut_and_joint_pair(
        arm_right, torso, [right_x, center_y, shoulder_z], [-1, 0, 0], shoulder_r, is_ball_joint, clearance_mm
    )

    # 5. Explode parts slightly so you can clearly see the joints in 3D Preview & print without fusing
    gap = height * 0.08
    head.apply_translation([0, 0, gap * 1.2])
    legs.apply_translation([0, 0, -gap * 0.5])
    arm_left.apply_translation([-gap * 1.1, 0, 0])
    arm_right.apply_translation([gap * 1.1, 0, 0])

    assembled_kit = trimesh.util.concatenate([head, torso, arm_left, arm_right, legs])
    min_z = assembled_kit.bounds[0][2]
    assembled_kit.apply_translation([0, 0, -min_z])
    return assembled_kit

# --- iPhone UI Controls ---
uploaded_file = st.file_uploader(
    "1. Tap to Snap Photo or Pick from iPhone Camera Roll",
    type=["png", "jpg", "jpeg", "webp"]
)

scale_label = st.selectbox(
    "2. Choose Library-Friendly Figure Scale",
    options=list(LIBRARY_SCALES.keys()),
    index=2
)

joint_mode = st.radio(
    "3. Choose Articulation / Joint Style",
    options=JOINT_MODES,
    index=0
)

col1, col2 = st.columns(2)
with col1:
    add_base = st.checkbox("Weld Flat Stand Base", value=True)
with col2:
    clearance = st.slider("Joint Clearance (mm)", 0.15, 0.40, 0.25, 0.05, help="0.25mm is ideal for library PLA printers so joints snap tight without fusing.")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="2D Character Reference", width=220)

    if st.button("🚀 Build Printable .STL Figure", type="primary", use_container_width=True):
        with st.spinner("Generating 3D mesh, carving joints & prepping for library printer..."):
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(temp_img.name, "PNG")

            engine_used = "Hunyuan3D-2 Cloud AI"
            try:
                glb_path = generate_mesh_hunyuan(temp_img.name)
                mesh = trimesh.load(glb_path, force="mesh")
            except Exception:
                mesh = build_fallback_mannequin()
                engine_used = "Test Mannequin (Free GPU queue busy — add HF_TOKEN in Secrets)"

            # Clean & orient upright
            mesh.fill_holes()
            mesh.fix_normals()
            min_z = mesh.bounds[0][2]
            mesh.apply_translation([0, 0, -min_z])

            # Scale to exact mm height
            target_mm = LIBRARY_SCALES[scale_label]
            mesh.apply_scale(target_mm / mesh.extents[2])

            # Apply Articulation Mode
            if "Ball-and-Socket" in joint_mode:
                mesh = articulate_figure(mesh, is_ball_joint=True, clearance_mm=clearance)
            elif "Swivel Pegs" in joint_mode:
                mesh = articulate_figure(mesh, is_ball_joint=False, clearance_mm=clearance)

            # Optional Collector Base under feet
            if add_base:
                radius = max(mesh.extents[0], mesh.extents[1]) * 0.42
                base_h = 3.2
                base = trimesh.creation.cylinder(radius=radius, height=base_h, sections=72)
                base.apply_translation([0, 0, base_h / 2.0])
                mesh.apply_translation([0, 0, base_h - 0.8])
                mesh = trimesh.util.concatenate([mesh, base])

            # Estimate PLA Filament weight
            volume_cm3 = abs(mesh.volume) / 1000.0
            est_grams = round(max(volume_cm3 * 1.24 * 0.28, 15.0), 1)

            glb_preview_bytes = mesh.export(file_type="glb")
            stl_bytes = mesh.export(file_type="stl")
            clean_tag = scale_label.split("-inch")[0].replace(".", "_") + "in"
            file_name = f"poseable_figure_{clean_tag}.stl"

        st.success(f"Done! ({engine_used})")

        st.subheader("🔄 Touch & Drag to Inspect 3D Figure & Joints")
        st.caption("Pinch with two fingers to zoom in on the separated ball-pegs and torso sockets.")
        render_3d_preview_on_iphone(glb_preview_bytes)

        st.info(
            f"**Library Slicer Checklist:**\n"
            f"- **Articulation:** {joint_mode}\n"
            f"- **Joint Tolerance:** {clearance} mm gap\n"
            f"- **Est. PLA Filament:** ~{est_grams} grams\n"
            f"- **Tip for Maker Lab:** In the library slicer (Bambu/Prusa), right-click the model and select **'Split to Objects' -> 'Auto-Orient'** so each limb lays flat on the print bed for maximum peg strength!"
        )

        st.download_button(
            label=f"📥 Save {file_name} to iPhone Files",
            data=stl_bytes,
            file_name=file_name,
            mime="application/sla",
            type="primary",
            use_container_width=True
        )
