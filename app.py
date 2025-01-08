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

    st.markdown("""
    **2. Teori Relativitas Umum dan Gelombang Gravitasi**
    
    Teori Relativitas Umum, yang dikemukakan oleh Albert Einstein pada tahun 1915, merupakan fondasi utama dalam pemahaman modern tentang gravitasi. Teori ini menggantikan konsep gravitasi Newton yang sebelumnya dianggap sebagai gaya tarik-menarik antara dua massa. Sebaliknya, Relativitas Umum menyatakan bahwa gravitasi adalah akibat dari kelengkungan ruang-waktu yang disebabkan oleh massa dan energi. Dalam kerangka ini, ruang dan waktu tidak lagi dianggap sebagai entitas terpisah, melainkan sebagai satu kesatuan empat dimensi yang dapat dibelokkan oleh keberadaan massa dan energi.
    
    Persamaan Medan Einstein adalah inti dari Teori Relativitas Umum, yang secara matematis menggambarkan hubungan antara distribusi massa-energi dan kelengkungan ruang-waktu. Solusi dari persamaan ini dalam kondisi tertentu menunjukkan adanya gelombang gravitasi yang merambat melalui ruang-waktu. Gelombang gravitasi ini memiliki dua jenis polarisasi utama, yaitu polarisasi "plus" dan "cross", yang menggambarkan cara ruang-waktu melengkung dan meregang dalam dua arah yang saling tegak lurus. Polarisasi ini menentukan bagaimana gelombang gravitasi mempengaruhi objek-objek yang dilalui, menyebabkan perubahan bentuk ruang-waktu yang sangat kecil namun terukur.
    
    Pembentukan gelombang gravitasi terjadi ketika massa besar mengalami akselerasi atau perubahan gerak yang signifikan. Misalnya, saat dua lubang hitam mengorbit dan akhirnya bergabung, atau ketika bintang neutron bertabrakan dan membentuk objek yang lebih besar. Gangguan ini menciptakan fluktuasi dalam ruang-waktu yang merambat keluar dari sumbernya dengan kecepatan cahaya. Persamaan Einstein yang telah dilinierkan untuk menggambarkan gelombang gravitasi yang lemah menunjukkan bahwa amplitudo gelombang ini berkurang seiring jarak dari sumber, memungkinkan deteksi gelombang gravitasi dari peristiwa kosmik yang sangat jauh.
    """)

    st.markdown("""
    **3. Sumber Gelombang Gravitasi**
    
    Gelombang gravitasi dihasilkan oleh peristiwa kosmik yang melibatkan perubahan massa atau akselerasi yang sangat besar. Beberapa sumber utama gelombang gravitasi meliputi penggabungan lubang hitam, penggabungan bintang neutron, supernova, dan gelombang gravitasi primordial.
    
    Penggabungan lubang hitam adalah salah satu sumber paling kuat dari gelombang gravitasi. Ketika dua lubang hitam yang berputar mengelilingi satu sama lain semakin mendekat hingga akhirnya bergabung menjadi satu lubang hitam yang lebih besar, mereka menghasilkan gelombang gravitasi yang sangat kuat. Deteksi pertama gelombang gravitasi oleh LIGO pada tahun 2015, yang dikenal sebagai GW150914, adalah hasil dari penggabungan dua lubang hitam ini. Gelombang gravitasi yang dihasilkan selama proses penggabungan ini membawa informasi tentang massa, spin, dan jarak lubang hitam yang terlibat, memberikan wawasan mendalam tentang dinamika objek-objek tersebut.
    
    Penggabungan bintang neutron juga merupakan sumber signifikan gelombang gravitasi. Bintang neutron adalah sisa-sisa supernova yang sangat padat, dan ketika dua bintang neutron saling mendekat hingga bertabrakan, mereka menghasilkan gelombang gravitasi serta ledakan elektromagnetik seperti kilonova. Deteksi GW170817 adalah contoh penting dari penggabungan bintang neutron, yang juga diikuti oleh pengamatan elektromagnetik, menunjukkan pentingnya deteksi multimessenger dalam memahami fenomena kosmik.
    
    Supernova, yaitu ledakan bintang masif yang menandai akhir hidup bintang tersebut, juga dapat menghasilkan gelombang gravitasi. Meskipun gelombang gravitasi dari supernova umumnya lebih lemah dibandingkan dengan penggabungan lubang hitam atau bintang neutron, mereka tetap penting untuk memahami mekanisme ledakan supernova dan sifat materi di bawah tekanan ekstrem.
    
    Selain itu, gelombang gravitasi primordial dihasilkan selama periode inflasi kosmik di alam semesta sangat awal. Gelombang gravitasi ini memberikan wawasan tentang kondisi dan proses di alam semesta awal, yang sulit diakses melalui metode pengamatan lainnya. Studi tentang gelombang gravitasi primordial dapat membantu menjawab pertanyaan mendasar tentang asal-usul dan evolusi alam semesta.
    """)

    st.markdown("""
    **4. Metode Deteksi Gelombang Gravitasi**
    
    Deteksi gelombang gravitasi merupakan tantangan teknis yang besar karena amplitudo gelombang ini sangat kecil ketika mencapai Bumi. Metode utama yang digunakan untuk mendeteksi gelombang gravitasi adalah interferometri laser, yang melibatkan penggunaan interferometer yang sangat sensitif untuk mengukur perubahan jarak yang disebabkan oleh gelombang gravitasi.
    
    Interferometer laser seperti LIGO (Laser Interferometer Gravitational-Wave Observatory), Virgo, dan KAGRA adalah detektor utama yang digunakan untuk mendeteksi gelombang gravitasi. Prinsip kerja interferometer laser melibatkan dua atau lebih lengan panjang yang saling tegak lurus, biasanya puluhan ribu meter. Sinar laser dipancarkan ke ujung masing-masing lengan, dipantulkan kembali oleh cermin, dan kemudian digabungkan kembali. Ketika gelombang gravitasi melintasi Bumi, mereka menyebabkan perubahan yang sangat kecil dalam panjang lengan interferometer, yang kemudian mempengaruhi pola interferensi laser. Detektor ini mampu mengukur perubahan panjang yang sangat kecil, bahkan sampai pada orde satu bagian per sepuluh ribu triliun dari panjang lengan interferometer.
    
    Kepekaan interferometer sangat dipengaruhi oleh berbagai sumber kebisingan, termasuk getaran seismik, kebisingan termal pada material detektor, dan fluktuasi instrumental. Untuk mengatasi tantangan ini, teknologi canggih dan teknik pengurangan kebisingan diterapkan. Misalnya, detektor dipasang di lokasi yang relatif tenang secara seismik, dan dilengkapi dengan sistem isolasi yang dapat mengurangi getaran eksternal. Selain itu, pemrosesan sinyal yang canggih digunakan untuk memisahkan sinyal gelombang gravitasi dari kebisingan latar belakang, memungkinkan deteksi yang lebih akurat.
    
    Deteksi multimessenger merupakan pendekatan yang menggabungkan pengamatan gelombang gravitasi dengan pengamatan gelombang elektromagnetik, neutrino, dan partikel lainnya. Pendekatan ini memungkinkan pemetaan sumber yang lebih akurat dan memberikan pemahaman yang lebih mendalam tentang peristiwa kosmik yang diamati. Misalnya, deteksi GW170817 tidak hanya melibatkan pengamatan gelombang gravitasi oleh LIGO dan Virgo, tetapi juga pengamatan kilonova melalui teleskop elektromagnetik, yang memberikan gambaran komprehensif tentang penggabungan bintang neutron.
    
    Selain interferometer darat, detektor ruang angkasa seperti Laser Interferometer Space Antenna (LISA) sedang dikembangkan untuk mendeteksi gelombang gravitasi pada frekuensi yang lebih rendah. LISA akan mengamati gelombang gravitasi yang dihasilkan oleh sumber-sumber yang berbeda, seperti penggabungan lubang hitam supermasif, dengan menghindari kebisingan dari Bumi. Detektor ruang angkasa ini diharapkan dapat memperluas jangkauan deteksi gelombang gravitasi dan membuka peluang baru dalam penelitian kosmik.
    """)

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
