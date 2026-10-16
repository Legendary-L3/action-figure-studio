import base64
import tempfile
import trimesh
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from gradio_client import Client, handle_file

st.set_page_config(page_title="Action Figure 3D Studio", page_icon="🦸", layout="centered")

st.title("🦸 Action Figure 3D Studio")
st.caption("iPhone-to-Library Edition • Sized for PLA 3D Printers")

LIBRARY_SCALES = {
    "3.75-inch Vintage Scale (95mm) | ~1.5 - 2.5 hrs": 95.25,
    "5-inch Classic Scale (127mm) | ~3 - 4.5 hrs": 127.0,
    "6-inch Collector Scale (152mm) | ~5 - 7 hrs": 152.4,
    "7-inch Deluxe Scale (178mm) | ~6.5 - 8 hrs": 177.8,
}

def render_3d_preview_on_iphone(glb_bytes: bytes):
    """Renders an interactive touch-enabled 3D viewer directly in iPhone Safari."""
    b64_glb = base64.b64encode(glb_bytes).decode("utf-8")
    html_code = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{b64_glb}"
        alt="3D Action Figure Preview"
        auto-rotate
        camera-controls
        shadow-intensity="1"
        style="width: 100%; height: 360px; background-color: #1e1e24; border-radius: 12px;">
    </model-viewer>
    """
    components.html(html_code, height=375)

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
    """Fallback test model so you can test scaling & STL downloads anytime."""
    torso = trimesh.creation.capsule(height=0.65, radius=0.18)
    torso.apply_translation([0, 0, 0.75])
    head = trimesh.creation.icosphere(radius=0.15)
    head.apply_translation([0, 0, 1.32])
    leg_l = trimesh.creation.capsule(height=0.55, radius=0.08)
    leg_l.apply_translation([-0.1, 0, 0.28])
    leg_r = trimesh.creation.capsule(height=0.55, radius=0.08)
    leg_r.apply_translation([0.1, 0, 0.28])
    return trimesh.util.concatenate([torso, head, leg_l, leg_r])

# --- iPhone UI ---
uploaded_file = st.file_uploader(
    "1. Tap to Snap Photo or Pick from iPhone Camera Roll",
    type=["png", "jpg", "jpeg", "webp"]
)

scale_label = st.selectbox(
    "2. Choose Library-Friendly Figure Scale",
    options=list(LIBRARY_SCALES.keys()),
    index=1
)

col1, col2 = st.columns(2)
with col1:
    add_base = st.checkbox("Weld Flat Stand Base", value=True)
with col2:
    base_thick = st.slider("Base Thickness (mm)", 2.0, 5.0, 3.2, 0.2)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="2D Character Reference", width=220)

    if st.button("🚀 Build Printable .STL Figure", type="primary", use_container_width=True):
        with st.spinner("Generating 3D mesh & prepping for library printer (~35-45 sec)..."):
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(temp_img.name, "PNG")

            engine_used = "Hunyuan3D-2 Cloud AI"
            try:
                glb_path = generate_mesh_hunyuan(temp_img.name)
                mesh = trimesh.load(glb_path, force="mesh")
            except Exception:
                mesh = build_fallback_mannequin()
                engine_used = "Test Mannequin (Free GPU queue busy — add free HF_TOKEN in Streamlit Secrets)"

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
                radius = max(mesh.extents[0], mesh.extents[1]) * 0.46
                base = trimesh.creation.cylinder(radius=radius, height=base_thick, sections=72)
                base.apply_translation([0, 0, base_thick / 2.0])
                mesh.apply_translation([0, 0, base_thick - 0.8])
                mesh = trimesh.util.concatenate([mesh, base])

            # 4. Estimate PLA Filament weight
            volume_cm3 = abs(mesh.volume) / 1000.0
            est_grams = round(max(volume_cm3 * 1.24 * 0.28, 14.0), 1)

            # 5. Export both GLB (for iPhone 3D touch preview) and STL (for library slicer)
            glb_preview_bytes = mesh.export(file_type="glb")
            stl_bytes = mesh.export(file_type="stl")
            clean_tag = scale_label.split("-inch")[0].replace(".", "_") + "in"
            file_name = f"action_figure_{clean_tag}.stl"

        st.success(f"Model Ready! ({engine_used})")
        
        st.subheader("🔄 Touch & Drag to Inspect 3D Figure")
        render_3d_preview_on_iphone(glb_preview_bytes)

        st.info(
            f"**Library Slicer Checklist:**\n"
            f"- **Final Height:** {round(mesh.extents[2], 1)} mm\n"
            f"- **Est. PLA Filament:** ~{est_grams} grams\n"
            f"- **Settings:** 0.16mm layer height • 15% Gyroid infill • Tree Supports (Build Plate Only)"
        )

        st.download_button(
            label=f"📥 Save {file_name} to iPhone Files",
            data=stl_bytes,
            file_name=file_name,
            mime="application/sla",
            type="primary",
            use_container_width=True
        )
