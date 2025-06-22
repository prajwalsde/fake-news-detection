import streamlit as st
import pickle
from PIL import Image
import base64

# === Page Configuration ===
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# === Load Model ===
with open("model/model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# === Title and Description ===
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a news article or paragraph and check if it's Real or Fake using a trained Machine Learning model.</p>", unsafe_allow_html=True)

# === Input Area ===
st.markdown("### 🧾 Enter News Article:")
input_text = st.text_area("", height=200, placeholder="Start typing or paste your news content here...")

# === Predict Button ===
if st.button("🔍 Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing..."):
            transformed = vectorizer.transform([input_text])
            prediction = model.predict(transformed)[0]
        
        # === Result Message ===
        if prediction == 1:
            st.success("✅ This news looks **Real**.")
            st.balloons()
        else:
            st.error("❌ This news seems **Fake**.")
            st.markdown("> Always verify with credible sources before sharing!")

# === Footer ===
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em;'>Made with ❤️ by Prajwal Jadhav | Powered by ML</p>",
    unsafe_allow_html=True,
)
