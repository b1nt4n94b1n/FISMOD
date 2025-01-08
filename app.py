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

# Membuat menu navigasi di sidebar
page = st.sidebar.selectbox(
    "Navigasi",
    ("Materi Gelombang Gravitasi", "Model Prediksi")
)

# ---------------------------
# Bagian 2: Halaman Materi
# ---------------------------
if page == "Materi Gelombang Gravitasi":
    st.header("Materi Deteksi Gelombang Gravitasi Menggunakan Model CNN")
    
    st.markdown("""
    **1. Pengantar Gelombang Gravitasi**
    
    Gelombang gravitasi adalah fenomena kosmik yang menggambarkan riak-riak dalam struktur ruang-waktu yang dihasilkan oleh pergerakan massa besar atau akselerasi objek-objek astronomis yang sangat masif. Konsep ini pertama kali diusulkan oleh Albert Einstein pada tahun 1915 melalui Teori Relativitas Umum. Menurut teori ini, massa dan energi dapat membelokkan ruang-waktu di sekitarnya, dan ketika massa tersebut bergerak atau mengalami akselerasi, gangguan ini merambat sebagai gelombang gravitasi. Meskipun prediksi teoritisnya telah ada selama lebih dari satu abad, gelombang gravitasi baru dapat dideteksi secara langsung pada tahun 2015 oleh observatorium LIGO (Laser Interferometer Gravitational-Wave Observatory). Penemuan ini menandai tonggak penting dalam fisika modern, membuka jendela baru bagi para ilmuwan untuk mengamati dan memahami peristiwa kosmik yang tidak dapat dijangkau melalui pengamatan elektromagnetik tradisional seperti cahaya, radio, atau sinar-X.

    Signifikansi gelombang gravitasi terletak pada kemampuannya untuk memberikan informasi langsung dari sumber-sumber kosmik yang ekstrem, seperti penggabungan lubang hitam atau bintang neutron. Gelombang gravitasi membawa data yang tidak terdistorsi oleh materi antar bintang, memungkinkan pengamatan yang lebih murni dan akurat tentang peristiwa-peristiwa tersebut. Selain itu, deteksi gelombang gravitasi memungkinkan pengujian lebih lanjut terhadap Teori Relativitas Umum dan membuka peluang untuk mengungkap fenomena baru di alam semesta, yang sebelumnya tidak dapat dipahami hanya melalui pengamatan elektromagnetik.
    """)

    # Sisanya tetap sama, bagian ini tidak berubah.

# ---------------------------
# Bagian 3: Halaman Model
# ---------------------------
elif page == "Model Prediksi":
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

    # Load model
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

    # Komponen file uploader dari Streamlit
    uploaded_file = st.file_uploader("Unggah gambar di sini (format: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    # Fungsi untuk melakukan prediksi
    def predict_image(uploaded_image):
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize((img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Sesuai VGG16

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
