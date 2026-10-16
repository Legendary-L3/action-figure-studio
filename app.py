import os
import base64
import tempfile
import trimesh
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from gradio_client import Client, handle_file

st.set_page_config(page_title="Action Figure 3D Studio", page_icon="🦸", layout="centered")

st.title("🦸 Action Figure 3D Studio")
st.caption("iPhone-to-Library Edition • Real 3D AI Sculpt + Live Joint Calibration")

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
if "engine_name" not in st.session_state:
    st.session_state.engine_name = ""

def render_3d_preview_on_iphone(glb_bytes: bytes):
    """Renders an interactive touch-enabled 3D turntable in iPhone Safari."""
    b64_glb = base64.b64encode(glb_bytes).decode("utf-8")
    html_code = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{b64_glb}"
        alt="3D Action Figure Preview"
        camera-controls
        auto-rotate
        shadow-intensity="1"
        exposure="1.1"
        style="width: 100%; height: 420px; background-color: #1e1e24; border-radius: 12px;">
    </model-viewer>
    """
    components.html(html_code, height=435)

def extract_glb_path(result) -> str:
    """Safely extracts the file path from various Gradio Client return formats."""
    if isinstance(result, str) and os.path.exists(result):
        return result
    if isinstance(result, dict):
        for k in ["value", "path", "name", "mesh"]:
            if k in result and isinstance(result[k], str) and os.path.exists(result[k]):
                return result[k]
    if isinstance(result, (list, tuple)):
        for item in result:
            found = extract_glb_path(item)
            if found:
                return found
    raise ValueError(f"Could not find 3D file path in API response: {result}")

def generate_real_3d_mesh(image_path: str) -> tuple[str, str]:
    """
    Tries the top free Image-to-3D Spaces on Hugging Face with full parameter signatures.
    Returns (glb_file_path, engine_used_label).
    """
    hf_token = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None
    errors = []

    # 1. Try Tencent Hunyuan3D-2.1 / Hunyuan3D-2 (High-detail character sculpt)
    for space_id in ["tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2", "tencent/Hunyuan3D-2mv"]:
        try:
            client = Client(space_id, hf_token=hf_token)
            for api_ep in ["/shape_generation", "/generation_all"]:
                try:
                    res = client.predict(
                        caption="",
                        image=handle_file(image_path),
                        mv_image_front=None,
                        mv_image_back=None,
                        mv_image_left=None,
                        mv_image_right=None,
                        steps=30,
                        guidance_scale=5.0,
                        seed=1234,
                        octree_resolution=256,
                        check_box_rembg=True,
                        num_
