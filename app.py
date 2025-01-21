import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import gdown

# ---------------------------
# Bagian 1: Setup Aplikasi
# ---------------------------
st.set_page_config(
    page_title="Aplikasi Deteksi Gelombang Gravitasi",
    layout="centered"
)

# Membuat judul utama
st.title("Aplikasi Deteksi Gelombang Gravitasi")

# Membuat menu navigasi di sidebar (hanya Model Prediksi)
page = st.sidebar.selectbox(
    "Navigasi",
    ("Model Prediksi",)  # Hanya satu opsi
)

# ---------------------------
# Halaman Model Prediksi
# ---------------------------
if page == "Model Prediksi":
    st.header("Halaman Model Prediksi")
    st.write("""
    Silakan unggah gambar glitch gravitasional yang ingin Anda prediksi. 
    Model akan memberikan label prediksi dan probabilitas untuk masing-masing kelas.
    """)

    # ID file Google Drive (ganti sesuai dengan file model Anda)
    file_id = "1aPmVk3vej_sFjhSOpHrYFSqanxYw951M"  # Ganti dengan ID file model Anda
    model_path = "GravCNN_test.h5"

    # Unduh model dari Google Drive jika belum ada
    if not os.path.exists(model_path):
        with st.spinner("Mengunduh model dari Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        st.success("Model berhasil diunduh.")
    else:
        st.info("Model sudah tersedia di direktori lokal.")

    # Fungsi memuat model
    @st.cache(allow_output_mutation=True)
    def load_model_from_path(path):
        return load_model(path)

    model = load_model_from_path(model_path)

    # Label kelas (diambil dari train_generator.class_indices)
    class_labels = {
        0: '1080Lines', 1: '1400Ripples', 2: 'Air_Compressor', 3: 'Blip',
        4: 'Chirp', 5: 'Extremely_Loud', 6: 'Helix', 7: 'Koi_Fish',
        8: 'Light_Modulation', 9: 'Low_Frequency_Burst', 10: 'Low_Frequency_Lines',
        11: 'No_Glitch', 12: 'None_of_the_Above', 13: 'Paired_Doves',
        14: 'Power_Line', 15: 'Repeating_Blips', 16: 'Scattered_Light',
        17: 'Scratchy', 18: 'Tomte', 19: 'Violin_Mode', 20: 'Wandering_Line',
        21: 'Whistle'
    }

    # Ukuran input gambar
    img_height = 224
    img_width = 224

    # Komponen file uploader di Streamlit
    uploaded_file = st.file_uploader("Unggah gambar di sini (format: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    # Fungsi untuk melakukan prediksi
    def predict_image(uploaded_image):
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize((img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Sesuai preprocessing VGG16

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        probabilities = predictions[0] * 100  # Probabilitas dalam persen

        return predicted_label, probabilities

    # Jika pengguna mengunggah file, tampilkan gambar & prediksi
    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

        # Tombol untuk prediksi
        if st.button("Prediksi Gambar"):
            label, probs = predict_image(uploaded_file)

            st.markdown(f"### Hasil Prediksi: **{label}**")
            st.write("Probabilitas untuk setiap kelas:")
            for class_name, probability in zip(class_labels.values(), probs):
                st.write(f"{class_name}: {probability:.2f}%")
