import streamlit as st
import streamlit.components.v1 as components


def about_dataset():

    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'Inter', sans-serif;
            background: transparent;
            color: #E5E7EB;
        }

        .section-card {
            background: #020617;
            border-radius: 22px;
            padding: 2.4rem;
            margin-bottom: 28px;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 20px 45px rgba(0,0,0,0.45);
        }

        .section-title {
            font-size: 26px;
            font-weight: 600;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-text {
            color: #9CA3AF;
            font-size: 15.5px;
            line-height: 1.75;
        }

        .pill {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            background: rgba(20,184,166,0.18);
            color: #5EEAD4;
            font-weight: 500;
            font-size: 13.5px;
            margin-right: 10px;
            margin-top: 16px;
            border: 1px solid rgba(94,234,212,0.25);
        }

        .pipeline-card {
            background: #020617;
            border-radius: 16px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 14px;
            border-left: 5px solid #14B8A6;
        }

        .pipeline-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .pipeline-desc {
            font-size: 14.5px;
            color: #9CA3AF;
            line-height: 1.65;
        }
        </style>
        </head>

        <body>

        <!-- ================= DATASET ================= -->
        <div class="section-card">
            <div class="section-title">📁 Tentang Dataset</div>

            <div class="section-text">
                Kualitas udara merupakan salah satu faktor penting yang
                memengaruhi kesehatan masyarakat, terutama di wilayah
                perkotaan dengan tingkat aktivitas tinggi seperti
                <b>DKI Jakarta</b>.
                <br><br>
                Dataset ini berisi data konsentrasi <b>polutan udara</b> yang
                dikumpulkan dari beberapa lokasi pemantauan di Jakarta,
                dan digunakan untuk <b>mengklasifikasikan kualitas udara</b>
                ke dalam beberapa kategori berdasarkan tingkat pencemarannya.
            </div>

            <div>
                <span class="pill">Baik</span>
                <span class="pill">Sedang</span>
                <span class="pill">Tidak Sehat</span>
            </div>
        </div>

        <!-- ================= FEATURE ================= -->
        <div class="section-card">
            <div class="section-title">🧪 Variabel dalam Dataset</div>

            <div class="section-text">
                Variabel independen yang digunakan dalam dataset ini
                merupakan parameter utama kualitas udara:
            </div>

            <ul class="section-text">
                <li><b>PM10</b> – Partikulat berdiameter &le; 10 &micro;m</li>
                <li><b>PM2.5</b> – Partikulat halus berdiameter &le; 2.5 &micro;m</li>
                <li><b>SO<sub>2</sub></b> – Sulfur Dioksida</li>
                <li><b>CO</b> – Karbon Monoksida</li>
                <li><b>O<sub>3</sub></b> – Ozon</li>
                <li><b>NO<sub>2</sub></b> – Nitrogen Dioksida</li>
            </ul>

            <div class="section-text">
                Target klasifikasi pada penelitian ini adalah
                <b>categori</b>, yaitu kategori kualitas udara.
            </div>
        </div>

        <!-- ================= ALGORITHM ================= -->
        <div class="section-card">
            <div class="section-title">🤖 Algoritma yang Digunakan</div>

            <div class="section-text">
                Algoritma utama yang digunakan adalah
                <b>Random Forest Classifier</b>,
                yaitu metode <i>ensemble learning</i>
                yang menggabungkan banyak pohon keputusan
                untuk meningkatkan akurasi dan stabilitas model.
            </div>

            <ul class="section-text">
                <li>Mampu menangani hubungan <b>non-linear</b></li>
                <li>Tahan terhadap <b>noise</b> dan overfitting</li>
                <li>Tidak memerlukan normalisasi data</li>
                <li>Cocok untuk dataset tabular berukuran kecil</li>
            </ul>
        </div>

        <!-- ================= PIPELINE ================= -->
        <div class="section-title">🧭 Alur Klasifikasi Random Forest</div>

        <div class="pipeline-card">
            <div class="pipeline-title">1. Input Data Polutan</div>
            <div class="pipeline-desc">
                Data PM10, PM2.5, SO2, CO, O3, dan NO2 digunakan sebagai fitur input.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">2. Bootstrap Sampling</div>
            <div class="pipeline-desc">
                Data diambil secara acak dengan pengembalian untuk tiap pohon.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">3. Pembangunan Pohon</div>
            <div class="pipeline-desc">
                Setiap pohon dibangun dari subset fitur acak.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">4. Voting Mayoritas</div>
            <div class="pipeline-desc">
                Hasil prediksi ditentukan dari suara terbanyak.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">5. Prediksi Akhir</div>
            <div class="pipeline-desc">
                Kategori kualitas udara ditentukan sebagai output akhir.
            </div>
        </div>

        </body>
        </html>
        """,
        height=1650,
        scrolling=True
    )
