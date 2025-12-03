# app.py — NutriVision (Groq Vision, optimized with safe compression)

import os
import io
import base64
import requests
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(
    page_title="NutriVision AI (Groq)",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🥗 NutriVision — AI-Powered Food & Calorie Analyzer")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------------
# Safe image compression
# -------------------------------
def prepare_image(uploaded_file):
    """
    Compress & resize image safely.
    Ensures base64 size stays under Groq limit.
    """
    img = Image.open(uploaded_file).convert("RGB")

    # resize to safe resolution
    img.thumbnail((900, 900))

    # compress
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)

    return Image.open(buf)


# -------------------------------
# Base64 encode
# -------------------------------
def encode_image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------
# API call
# -------------------------------
def analyze_food_image(prompt: str, image: Image.Image, temperature: float = 0.3) -> str:
    img_b64 = encode_image_to_b64(image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} ,
                ],
            }
        ],
        "max_tokens": 700,
        "temperature": temperature,
        "stream": False,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=120)

        if resp.status_code == 413:
            return "⚠️ The image is still too large. Try uploading a smaller image."

        if not resp.ok:
            try:
                return f"⚠️ API error {resp.status_code}: {resp.json()}"
            except Exception:
                resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ Error communicating with Groq API: {e}"


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Model Settings")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)

default_prompt = (
    "You are a professional nutritionist. Identify each visible food item in the image. "
    "Estimate approximate calories per item and provide a total like:\n"
    "1) Item — ~calories\n2) Item — ~calories\nTotal — ~calories\n"
    "If uncertain, mention assumptions."
)

user_prompt = st.text_area("Instruction to the AI", default_prompt, height=140)


# -------------------------------
# Image Upload
# -------------------------------
uploaded = st.file_uploader("Upload a meal photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
image = None

if uploaded:
    if uploaded.size > 4 * 1024 * 1024:
        st.error("❌ Image too large. Upload < 4MB.")
        st.stop()

    # --- prepare image safely ---
    image = prepare_image(uploaded)

    st.image(image, caption="Uploaded Image", use_column_width=True)


# -------------------------------
# Analyze Button
# -------------------------------
if st.button("🍽️ Analyze"):
    if image is None:
        st.warning("Please upload a meal image first.")
    else:
        with st.spinner("🔍 Analyzing image with Groq Vision..."):
            result = analyze_food_image(user_prompt, image, temperature)

        st.subheader("🧠 AI Analysis")
        st.write(result)


# -------------------------------
# Footer
# -------------------------------
st.caption("Built with ❤️ Jasmin — Powered by Groq Vision")
