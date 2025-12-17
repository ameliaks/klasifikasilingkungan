import streamlit as st
import pandas as pd
import numpy as np
import joblib


def prediction_app():
    st.title("🔮 Prediksi Kategori Kualitas Udara (Jakarta)")
    st.write("Masukkan data polutan untuk memprediksi **categori** menggunakan model Logistic Regression.")

    # =========================
    # 1) Load model + artifacts (HARUS sama dengan training)
    # =========================
    model = joblib.load("model_polusi.pkl")
    feature_names = joblib.load("model_features_polusi.pkl")
    le = joblib.load("label_encoder_polusi.pkl")
    scaler = joblib.load("scaler_polusi.pkl")
    cont_cols = joblib.load("continuous_columns_polusi.pkl")

    # =========================
    # 2) Ambil kolom asli dari dataset (untuk form)
    # =========================
    df_raw = pd.read_csv("C:/Users/lenovo/Downloads/machine learning 2/polusijakarta.csv")
    df_raw.columns = (
        df_raw.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    rename_map = {
        "pm2.5": "pm25",
        "pm_25": "pm25",
        "kategori": "categori",
        "category": "categori",
        "station": "location",
        "lokasi": "location",
        "wilayah": "location",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})

    target_col = "categori"
    if target_col not in df_raw.columns:
        st.error("Kolom target 'categori' tidak ditemukan di dataset.")
        st.stop()

    if "tanggal" not in df_raw.columns:
        st.error("Kolom 'tanggal' tidak ditemukan di dataset. Prediksi butuh tanggal untuk membuat fitur waktu.")
        st.stop()

    # parse tanggal untuk default
    df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], errors="coerce")

    polutan_cols = [c for c in ["pm10", "pm25", "so2", "co", "o3", "no2"] if c in df_raw.columns]
    if len(polutan_cols) == 0:
        st.error("Kolom polutan tidak ditemukan (pm10/pm25/so2/co/o3/no2).")
        st.stop()

    location_col = "location" if "location" in df_raw.columns else None
    critical_col = "critical" if "critical" in df_raw.columns else None

    st.write("### Input Data Pemantauan")
    st.caption("Form ini menyesuaikan dataset polusi (polutan + konteks) dan akan membentuk fitur turunan seperti saat training.")

    # =========================
    # 3) Form input user
    # =========================
    col1, col2, col3 = st.columns(3)

    user_data = {}

    # --- Polutan ---
    with col1:
        st.write("**🧪 Polutan**")
        for p in polutan_cols:
            series = df_raw[p] if p in df_raw.columns else pd.Series([0])
            default_val = float(series.median()) if series.notna().any() else 0.0
            min_val = float(series.min()) if series.notna().any() else 0.0
            max_val = float(series.max()) if series.notna().any() else (default_val + 100.0)

            user_data[p] = st.number_input(
                p.upper(),
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )

    # --- Tanggal (untuk fitur waktu) ---
    with col2:
        st.write("**📅 Waktu**")
        default_date = df_raw["tanggal"].dropna().max()
        if pd.isna(default_date):
            default_date = pd.to_datetime("2025-01-01")
        tanggal_in = st.date_input("Tanggal", value=default_date.date())

    # --- Konteks (opsional) ---
    with col3:
        st.write("**📍 Konteks (Opsional)**")
        if location_col:
            loc_opts = sorted(df_raw[location_col].dropna().astype(str).unique().tolist())
            user_data["location"] = st.selectbox("Location", loc_opts if loc_opts else ["Unknown"])
        if critical_col:
            crit_opts = sorted(df_raw[critical_col].dropna().astype(str).unique().tolist())
            user_data["critical"] = st.selectbox("Critical", crit_opts if crit_opts else ["Unknown"])

    # =========================
    # 4) Bentuk fitur turunan (HARUS sama seperti training)
    # =========================
    dt = pd.to_datetime(tanggal_in)

    # fitur waktu
    user_data["tahun"] = dt.year
    user_data["bulan"] = dt.month
    user_data["hari"] = dt.day
    user_data["hari_dalam_minggu"] = dt.dayofweek
    user_data["minggu_ke"] = int(dt.isocalendar().week)

    # fitur agregat polutan
    user_data["total_polutan"] = float(sum(user_data[p] for p in polutan_cols))
    user_data["mean_polutan"] = float(np.mean([user_data[p] for p in polutan_cols]))

    dom_num = float(max(user_data[p] for p in polutan_cols))
    user_data["dominance_ratio"] = float(dom_num / user_data["total_polutan"]) if user_data["total_polutan"] > 0 else 0.0

    # =========================
    # 5) Preprocess input -> harus sama seperti training
    # =========================
    user_df = pd.DataFrame([user_data])

    # one-hot untuk location/critical seperti training (drop_first=True)
    cat_cols = [c for c in ["location", "critical"] if c in user_df.columns]
    user_encoded = pd.get_dummies(user_df, columns=cat_cols, drop_first=True)

    # samakan kolom dengan training
    user_encoded = user_encoded.reindex(columns=feature_names, fill_value=0)

    # scaling numerik continuous yang dipakai saat training
    cont_exist = [c for c in cont_cols if c in user_encoded.columns]
    if len(cont_exist) > 0:
        user_encoded[cont_exist] = scaler.transform(user_encoded[cont_exist])

    # =========================
    # 6) Prediksi + interpretasi
    # =========================
    if st.button("Prediksi Kategori"):
        pred_idx = model.predict(user_encoded)[0]
        pred_label = le.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(user_encoded)[0]
        proba_df = pd.DataFrame({
            "Kategori": le.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)

        top_prob = float(proba_df.iloc[0]["Probabilitas"])

        st.write("### ✅ Hasil Prediksi")
        st.success(f"Prediksi Kategori Kualitas Udara: **{pred_label}**")

        st.write("### Probabilitas Tiap Kategori")
        st.dataframe(proba_df, use_container_width=True)

        st.metric("Probabilitas Prediksi Tertinggi", f"{top_prob*100:.2f}%")

        # indikator keyakinan
        if top_prob < 0.50:
            st.warning("⚠️ Model kurang yakin (probabilitas tertinggi < 50%). Pertimbangkan cek ulang input atau tambah data training.")
        elif top_prob < 0.70:
            st.info("ℹ️ Keyakinan model sedang (50–70%). Cocok untuk screening awal.")
        else:
            st.success("✅ Keyakinan model tinggi (≥ 70%).")

        st.write("---")
        st.write("### 🧾 Interpretasi Singkat")
        st.write(
            "- **Baik**: kualitas udara relatif aman.\n"
            "- **Sedang**: perlu kewaspadaan bagi kelompok sensitif.\n"
            "- **Tidak Sehat**: berisiko bagi kesehatan, kurangi aktivitas luar.\n"
            "- (Jika dataset punya label lain, interpretasi akan mengikuti label di data kamu.)"
        )

        st.write("---")
        st.write("### 📌 Rekomendasi Tindak Lanjut (Umum)")

        label_key = str(pred_label).strip().lower()

        if "tidak" in label_key or "unhealthy" in label_key:
            st.error("⚠️ Udara tergolong tidak sehat — disarankan tindakan pencegahan segera.")
            recs = [
                "Kurangi aktivitas luar ruangan, terutama anak-anak, lansia, dan penderita asma/ISPA.",
                "Gunakan masker (mis. KN95) saat berada di luar.",
                "Tutup ventilasi saat polusi tinggi, gunakan air purifier bila ada.",
                "Pantau gejala (batuk, sesak) dan konsultasi layanan kesehatan bila memburuk."
            ]
        elif "sedang" in label_key or "moderate" in label_key:
            st.warning("⚠️ Udara sedang — kelompok sensitif perlu lebih waspada.")
            recs = [
                "Kelompok sensitif sebaiknya membatasi aktivitas berat di luar.",
                "Pantau kondisi udara di jam puncak (pagi/sore).",
                "Pertimbangkan masker saat polusi meningkat."
            ]
        else:
            st.success("✅ Udara relatif baik — tetap pantau perubahan harian.")
            recs = [
                "Aktivitas luar ruangan umumnya aman.",
                "Tetap pantau tren polusi karena bisa berubah cepat.",
                "Jaga kebiasaan sehat (hidrasi, istirahat cukup)."
            ]

        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")

        st.caption(
            "Catatan: prediksi ini berbasis model machine learning dari dataset yang tersedia. "
            "Untuk keputusan kesehatan, tetap mengacu pada sumber resmi & kondisi individu."
        )


if __name__ == "__main__":
    prediction_app()
