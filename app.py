import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Deteksi Sarkasme Komentar TikTok")

text = st.text_area("Masukkan komentar:")

if st.button("Deteksi"):
    if text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label = "Sarkas" if pred == 1 else "Tidak Sarkas"
        st.success(f"Hasil Prediksi: **{label}**")
