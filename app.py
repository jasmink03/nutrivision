# app.py — NutriVision (Groq Vision, current models)

from dotenv import load_dotenv
import os, io, base64, requests
from PIL import Image
import streamlit as st

# -------------------------------
# Env / key
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
# App config
# -------------------------------
st.set_page_config(page_title="NutriVision AI (Groq)")
st.title("🥗 NutriVision — AI-Powered Food & Calorie Analyzer")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# ✅ Use a current Groq vision model:
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # or "meta-llama/llama-4-maverick-17b-128e-instruct"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

# -------------------------------
# Helpers
# -------------------------------
def encode_image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_food_image(prompt: str, image: Image.Image, temperature: float = 0.3) -> str:
    img_b64 = encode_image_to_b64(image)  # keep image < ~4MB b64 per Groq limits
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}" }},
                ],
            }
        ],
        "max_tokens": 700,
        "temperature": temperature,
        "stream": False,
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=120)
        if not resp.ok:
            try:
                return f"⚠️ API error {resp.status_code}: {resp.json()}"
            except Exception:
                resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error: {e}"

# -------------------------------
# UI
# -------------------------------
st.sidebar.header("Model Settings")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)

default_prompt = (
    "You are a professional nutritionist. Identify each visible food item in the image. "
    "Estimate approximate calories per item and provide a total in this format:\n"
    "1) Item — ~calories\n2) Item — ~calories\nTotal — ~calories\n"
    "If uncertain, state brief assumptions."
)
user_prompt = st.text_area("Instruction to the AI", default_prompt, height=140)

uploaded = st.file_uploader("Upload a meal photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
image = None
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

if st.button("🍽️ Analyze"):
    if image is None:
        st.warning("Please upload a meal image first.")
    else:
        with st.spinner("🔍 Analyzing image with Groq Vision..."):
            result = analyze_food_image(user_prompt, image, temperature)
        st.subheader("🧠 AI Analysis")
        st.write(result)


st.caption("Built with ❤️ Jasmin — Powered by Groq Vision")
