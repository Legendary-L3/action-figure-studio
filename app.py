import os
import tempfile
import trimesh
import numpy as np
import streamlit as st
from PIL import Image
from gradio_client import Client, handle_file

st.set_page_config(page_title="Action Figure Statue Studio", page_icon="🗿", layout="centered")

st.title("🗿 Action Figure Statue Studio")
st.caption("iPhone-to-Library Edition • Live Photo-to-3D Statue Studio")

LIBRARY_SCALES = {
    "3.75-inch Vintage Scale (95mm) | ~1.5 - 2.5 hrs": 95.25,
    "5-inch Classic Scale (127mm) | ~3 - 4.5 hrs": 127.0,
    "6-inch Collector Scale (152mm) | ~5 - 7 hrs": 152.4,
    "7-inch Deluxe Scale (178mm) | ~6.5 - 8 hrs": 177.8,
}

def generate_mesh_hunyuan(image_path: str) -> str:
    """Calls free Hugging Face Hunyuan3D-2 space remotely via gradio_client."""
    hf_token = st.secrets.get("HF_TOKEN", None) if "HF_TOKEN" in st.secrets else None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        
    client = Client("tencent/Hunyuan3D-2")
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

def build_test_mannequin() -> trimesh.Trimesh:
    """Offline backup mannequin if cloud queue is busy."""
    torso = trimesh.creation.capsule(height=0.55, radius=0.16)
    torso.apply_translation([0, 0, 0.78])
    head = trimesh.creation.icosphere(radius=0.14)
    head.apply_translation([0, 0, 1.28])
    arm_l = trimesh.creation.capsule(height=0.45, radius=0.065)
    arm_l.apply_translation([-0.26, 0, 0.80])
    arm_r = trimesh.creation.capsule(height=0.45, radius=0.065)
    arm_r.apply_translation([0.26, 0, 0.80])
    leg_l = trimesh.creation.capsule(height=0.50, radius=0.075)
    leg_l.apply_translation([-0.09, 0, 0.28])
    leg_r = trimesh.creation.capsule(height=0.50, radius=0.075)
    leg_r.apply_translation([0.09, 0, 0.28])
    mesh = trimesh.util.concatenate([torso, head, arm_l, arm_r, leg_l, leg_r])
    mesh.fill_holes()
    mesh.fix_normals()
    min_z = mesh.bounds[0][2]
    mesh.apply_translation([0, 0, -min_z])
    mesh.apply_scale(152.4 / mesh.extents[2])
    return mesh

# --- iPhone UI Controls ---
uploaded_file = st.file_uploader(
    "1. Tap to Snap Photo or Pick from iPhone Camera Roll",
    type=["png", "jpg", "jpeg", "webp"]
)

use_test_mode = st.checkbox("🧪 Fallback to Test Mannequin (Only if cloud queue is busy)", value=False)

scale_label = st.selectbox(
    "2. Choose Library-Friendly Statue Scale",
    options=list(LIBRARY_SCALES.keys()),
    index=2
)

col1, col2 = st.columns(2)
with col1:
    add_base = st.checkbox("Weld Flat Collector Base", value=True)
with col2:
    base_thick = st.slider("Base Thickness (mm)", 2.0, 5.0, 3.2, 0.2)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Character Photo", width=220)

    if st.button("🚀 Sculpt 3D Statue from Photo", type="primary", use_container_width=True):
        with st.spinner("AI is converting your photo into a 3D character mesh (~35-45 sec)..."):
            if use_test_mode:
                mesh = build_test_mannequin()
            else:
                try:
                    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(temp_img.name, "PNG")
                    glb_path = generate_mesh_hunyuan(temp_img.name)
                    mesh = trimesh.load(glb_path, force="mesh")
                except Exception as e:
                    st.warning(f"Cloud AI queue busy or error ({e}). Loading fallback mannequin.")
                    mesh = build_test_mannequin()

            # 1. Clean geometry & stand flat at Z=0
            mesh.fill_holes()
            mesh.fix_normals()
            min_z = mesh.bounds[0][2]
            mesh.apply_translation([0, 0, -min_z])

            # 2. Scale to exact millimeter height
            target_mm = LIBRARY_SCALES[scale_label]
            mesh.apply_scale(target_mm / mesh.extents[2])

            # 3. Weld flat circular base for PLA bed adhesion
            if add_base:
                radius = max(mesh.extents[0], mesh.extents[1]) * 0.45
                base = trimesh.creation.cylinder(radius=radius, height=base_thick, sections=72)
                base.apply_translation([0, 0, base_thick / 2.0])
                mesh.apply_translation([0, 0, base_thick - 0.8])
                mesh = trimesh.util.concatenate([mesh, base])

            # 4. Estimate PLA Filament weight
            volume_cm3 = abs(mesh.volume) / 1000.0
            est_grams = round(max(volume_cm3 * 1.24 * 0.28, 14.0), 1)

            stl_bytes = mesh.export(file_type="stl")
            clean_tag = scale_label.split("-inch")[0].replace(".", "_") + "in"
            file_name = f"character_statue_{clean_tag}.stl"

        st.success("Statue Sculpted from Your Photo!")
        st.info(
            f"**Library Slicer Specs:**\n"
            f"- **Final Height:** {round(mesh.extents[2], 1)} mm\n"
            f"- **Est. PLA Filament:** ~{est_grams} grams\n"
            f"- **Recommended Slicer Settings:** 0.16mm layer height • 15% Gyroid infill • Tree Supports (On Build Plate Only)"
        )

        st.download_button(
            label=f"📥 Save {file_name} to iPhone Files",
            data=stl_bytes,
            file_name=file_name,
            mime="application/sla",
            type="primary",
            use_container_width=True
        )
