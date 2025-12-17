import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# --- SMOTE: optional (biar ga error import) ---
IMBLEARN_OK = False
IMBLEARN_ERR = ""
SMOTE = None
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_OK = True
except Exception as e:
    IMBLEARN_OK = False
    IMBLEARN_ERR = str(e)
    SMOTE = None


def ml_model():
    st.title("🤖 Machine Learning — Prediksi Kategori Kualitas Udara (Jakarta)")
    st.caption("Target: **categori** | Model: **Multiclass Logistic Regression**")

    # =========================
    # 1) Load data
    # =========================
    df = pd.read_csv("C:/Users/lenovo/Downloads/machine learning 2/polusijakarta.csv")
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    # normalisasi nama kolom umum
    rename_map = {
        "pm2.5": "pm25",
        "pm_25": "pm25",
        "kategori": "categori",
        "category": "categori",
        "station": "location",
        "lokasi": "location",
        "wilayah": "location",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    target_col = "categori"
    if target_col not in df.columns:
        st.error("Kolom target 'categori' tidak ditemukan.")
        st.stop()

    if "tanggal" not in df.columns:
        st.error("Kolom 'tanggal' tidak ditemukan. Bagian ML butuh kolom tanggal untuk fitur waktu.")
        st.stop()

    # parse tanggal
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal", target_col]).reset_index(drop=True)

    # kolom polutan yang ada
    polutan_cols = [c for c in ["pm10", "pm25", "so2", "co", "o3", "no2"] if c in df.columns]
    if len(polutan_cols) == 0:
        st.error("Kolom polutan tidak ditemukan (pm10/pm25/so2/co/o3/no2).")
        st.stop()

    location_col = "location" if "location" in df.columns else None
    critical_col = "critical" if "critical" in df.columns else None
    max_col = "max" if "max" in df.columns else None

    st.write("**Dataset (preview)**")
    st.dataframe(df.head(10), use_container_width=True)

    # =========================
    # 2) Feature Engineering (biar variabel makin kaya)
    # =========================
    st.write("### 1. Feature Engineering (Fitur Turunan)")

    df_feat = df.copy()
    df_feat["tahun"] = df_feat["tanggal"].dt.year
    df_feat["bulan"] = df_feat["tanggal"].dt.month
    df_feat["hari"] = df_feat["tanggal"].dt.day
    df_feat["hari_dalam_minggu"] = df_feat["tanggal"].dt.dayofweek  # 0=Mon
    df_feat["minggu_ke"] = df_feat["tanggal"].dt.isocalendar().week.astype(int)

    # fitur agregat polutan
    df_feat["total_polutan"] = df_feat[polutan_cols].sum(axis=1)
    df_feat["mean_polutan"] = df_feat[polutan_cols].mean(axis=1)

    # dominance ratio pakai max kalau tersedia, kalau tidak, pakai max dari polutan
    if max_col and max_col in df_feat.columns:
        df_feat["dominance_ratio"] = np.where(df_feat["total_polutan"] > 0, df_feat["max"] / df_feat["total_polutan"], np.nan)
    else:
        df_feat["dominance_ratio"] = np.where(df_feat["total_polutan"] > 0, df_feat[polutan_cols].max(axis=1) / df_feat["total_polutan"], np.nan)

    # isi NaN dominance_ratio
    df_feat["dominance_ratio"] = df_feat["dominance_ratio"].fillna(df_feat["dominance_ratio"].median())

    st.info("Fitur dibuat: tahun, bulan, hari, hari_dalam_minggu, minggu_ke, total_polutan, mean_polutan, dominance_ratio.")

    # =========================
    # 3) Tentukan fitur numerik & kategorik
    # =========================
    numeric_cols = polutan_cols + ["tahun", "bulan", "hari", "hari_dalam_minggu", "minggu_ke",
                                  "total_polutan", "mean_polutan", "dominance_ratio"]

    cat_cols = []
    if location_col:
        cat_cols.append(location_col)
    if critical_col:
        cat_cols.append(critical_col)

    # pastikan kolom benar-benar ada
    numeric_cols = [c for c in numeric_cols if c in df_feat.columns]
    cat_cols = [c for c in cat_cols if c in df_feat.columns]

    # =========================
    # 4) Missing value handling + One-hot
    # =========================
    st.write("### 2. Encoding & Missing Value")

    df_model = df_feat.copy()

    for c in numeric_cols:
        df_model[c] = df_model[c].fillna(df_model[c].median())

    for c in cat_cols:
        mode_val = df_model[c].mode(dropna=True)
        df_model[c] = df_model[c].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown").astype(str)

    # encode target
    le = LabelEncoder()
    y = le.fit_transform(df_model[target_col].astype(str))
    class_names = list(le.classes_)

    # one-hot fitur kategorik
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    # buang kolom yang tidak dipakai sebagai fitur
    drop_cols = [target_col, "tanggal"]
    X = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors="ignore")

    st.write(f"Jumlah fitur setelah encoding: **{X.shape[1]} fitur**")

    # =========================
    # 5) Scaling MinMax untuk numeric
    # =========================
    st.write("### 3. Normalisasi (MinMaxScaler)")

    scaler = MinMaxScaler()
    cont_exist = [c for c in numeric_cols if c in X.columns]

    X_before = X.copy()
    if len(cont_exist) > 0:
        X[cont_exist] = scaler.fit_transform(X[cont_exist])

        # visual ringkas: pilih 2 fitur numerik untuk lihat before/after
        pick = cont_exist[:2] if len(cont_exist) >= 2 else cont_exist
        colA, colB = st.columns(2)

        with colA:
            st.write("**Sebelum Normalisasi (Density)**")
            for col in pick:
                chart = (
                    alt.Chart(pd.DataFrame({col: X_before[col]}))
                    .transform_density(col, as_=[col, "density"])
                    .mark_area(opacity=0.5)
                    .encode(x=alt.X(f"{col}:Q"), y=alt.Y("density:Q"))
                    .properties(height=220, title=f"{col}")
                )
                st.altair_chart(chart, use_container_width=True)

        with colB:
            st.write("**Sesudah Normalisasi (Density)**")
            for col in pick:
                chart = (
                    alt.Chart(pd.DataFrame({col: X[col]}))
                    .transform_density(col, as_=[col, "density"])
                    .mark_area(opacity=0.5)
                    .encode(x=alt.X(f"{col}:Q"), y=alt.Y("density:Q"))
                    .properties(height=220, title=f"{col}")
                )
                st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tidak ada kolom numerik yang bisa dinormalisasi.")

    # =========================
    # 6) Train–Test Split
    # =========================
    st.write("### 4. Train–Test Split")

    # kalau data terlalu sedikit, stratify kadang gagal
    stratify_ok = True
    try:
        _ = np.unique(y, return_counts=True)
        # minimal 2 per kelas agar stratify aman
        if np.min(_[1]) < 2:
            stratify_ok = False
    except Exception:
        stratify_ok = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if stratify_ok else None
    )
    st.write(f"X_train: **{len(X_train)}** | X_test: **{len(X_test)}**")
    if not stratify_ok:
        st.warning("Stratify dimatikan karena beberapa kelas terlalu sedikit. Evaluasi bisa kurang stabil.")

    # =========================
    # 7) Handling Imbalance (SMOTE optional)
    # =========================
    st.write("### 5. Handling Imbalance Class")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribusi kelas (train sebelum balancing)**")
        uniq, cnt = np.unique(y_train, return_counts=True)
        before_counts = dict(zip(uniq, cnt))
        for k, name in enumerate(class_names):
            st.metric(label=name, value=int(before_counts.get(k, 0)))

    use_smote = False
    X_train_bal, y_train_bal = X_train, y_train

    if IMBLEARN_OK:
        try:
            sm = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
            use_smote = True
        except Exception as e:
            use_smote = False
            st.warning(f"SMOTE gagal dijalankan. Model akan pakai class_weight='balanced'. Detail: {e}")
    else:
        st.warning("SMOTE tidak tersedia (imblearn tidak terbaca). Model akan pakai class_weight='balanced'.")
        st.caption(f"Detail import: {IMBLEARN_ERR}")

    with col2:
        if use_smote:
            st.write("**Distribusi kelas (setelah SMOTE)**")
            uniq2, cnt2 = np.unique(y_train_bal, return_counts=True)
            after_counts = dict(zip(uniq2, cnt2))
            for k, name in enumerate(class_names):
                st.metric(label=name, value=int(after_counts.get(k, 0)))
        else:
            st.info("Balancing dilakukan via class_weight='balanced' (tanpa SMOTE).")

    # =========================
    # 8) Training model
    # =========================
    st.write("### 6. Training Model (Multiclass Logistic Regression)")

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=5000,
        class_weight=None if use_smote else "balanced"
    )
    model.fit(X_train_bal, y_train_bal)

    train_acc = model.score(X_train_bal, y_train_bal)
    st.write("Akurasi Training =", f"**{train_acc*100:.2f}%**")

    # Top fitur paling berpengaruh
    st.write("**Top 15 fitur paling berpengaruh (rata-rata |koefisien|)**")
    abs_mean = np.abs(model.coef_).mean(axis=0)
    top_idx = np.argsort(abs_mean)[::-1][:15]
    top_df = pd.DataFrame({
        "Feature": X.columns[top_idx],
        "Avg |Coefficient|": abs_mean[top_idx]
    })
    st.dataframe(top_df, use_container_width=True)

    # =========================
    # 9) Evaluasi
    # =========================
    st.write("### 7. Evaluasi Model")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC AUC multiclass (kadang gagal kalau kelas di test tidak lengkap)
    auc = np.nan
    try:
        proba = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
        if y_test_bin.shape[1] == proba.shape[1]:
            auc = roc_auc_score(y_test_bin, proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = np.nan

    colM1, colM2 = st.columns([2, 1])

    with colM1:
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names).reset_index().melt("index")
        cm_df.columns = ["Actual", "Predicted", "Count"]

        heat = (
            alt.Chart(cm_df).mark_rect()
            .encode(
                x=alt.X("Predicted:N", title="Predicted"),
                y=alt.Y("Actual:N", title="Actual"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Actual", "Predicted", "Count"]
            )
            .properties(height=360, title="Confusion Matrix")
        )
        text = alt.Chart(cm_df).mark_text(color="black").encode(
            x="Predicted:N", y="Actual:N", text="Count:Q"
        )
        st.altair_chart(heat + text, use_container_width=True)

    with colM2:
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("Precision (weighted)", f"{prec_w*100:.2f}%")
        st.metric("Recall (weighted)", f"{rec_w*100:.2f}%")
        st.metric("F1 Score (weighted)", f"{f1_w*100:.2f}%")
        st.metric("ROC AUC (weighted, OVR)", "-" if np.isnan(auc) else f"{auc*100:.2f}%")

    # =========================
    # 10) Simpan artifacts
    # =========================
    st.write("### 8. Simpan Model & Artefak")

    joblib.dump(model, "model_polusi.pkl")
    joblib.dump(list(X.columns), "model_features_polusi.pkl")
    joblib.dump(le, "label_encoder_polusi.pkl")
    joblib.dump(cont_exist, "continuous_columns_polusi.pkl")
    joblib.dump(scaler, "scaler_polusi.pkl")

    st.success("Model & file pendukung berhasil disimpan: model_polusi.pkl + artifacts lainnya.")

    # =========================
    # 11) Form Prediksi (1 data)
    # =========================
    st.write("### 9. Prediksi Kategori (Input Manual)")

    with st.expander("🔮 Coba Prediksi dengan Input Manual", expanded=True):
        colA, colB, colC = st.columns(3)

        # input polutan
        inputs = {}
        with colA:
            st.write("**Polutan**")
            for p in polutan_cols:
                inputs[p] = st.number_input(p.upper(), value=float(df[p].median()) if p in df.columns else 0.0)

        with colB:
            st.write("**Waktu**")
            tanggal_in = st.date_input("Tanggal", value=df["tanggal"].max().date())
            # derive time features
            dt = pd.to_datetime(tanggal_in)
            inputs["tahun"] = dt.year
            inputs["bulan"] = dt.month
            inputs["hari"] = dt.day
            inputs["hari_dalam_minggu"] = dt.dayofweek
            inputs["minggu_ke"] = int(dt.isocalendar().week)

            # agregat
            inputs["total_polutan"] = float(sum(inputs[p] for p in polutan_cols))
            inputs["mean_polutan"] = float(np.mean([inputs[p] for p in polutan_cols]))
            # dominance ratio
            dom_num = float(max(inputs[p] for p in polutan_cols))
            inputs["dominance_ratio"] = float(dom_num / inputs["total_polutan"]) if inputs["total_polutan"] > 0 else 0.0

        # input kategori tambahan (location/critical) jika ada
        extra_cats = {}
        with colC:
            st.write("**Konteks (opsional)**")
            if location_col:
                extra_cats["location"] = st.selectbox("Location", sorted(df[location_col].astype(str).unique().tolist()))
            if critical_col:
                extra_cats["critical"] = st.selectbox("Critical", sorted(df[critical_col].astype(str).unique().tolist()))

        if st.button("Prediksi"):
            # bangun 1-row dataframe sesuai pipeline
            row = inputs.copy()

            # gabungkan kategori tambahan sesuai schema sebelum one-hot (drop_first=True)
            base_row = pd.DataFrame([row])

            # tambahkan kolom kategorik agar bisa di-dummy sesuai training
            for c in ["location", "critical"]:
                if c in extra_cats:
                    base_row[c] = str(extra_cats[c])

            # one-hot seperti training (drop_first=True)
            base_row = pd.get_dummies(base_row, columns=[c for c in ["location", "critical"] if c in base_row.columns], drop_first=True)

            # pastikan urutan/kolom sama persis dengan training
            feat_names = joblib.load("model_features_polusi.pkl")
            for col in feat_names:
                if col not in base_row.columns:
                    base_row[col] = 0
            base_row = base_row[feat_names]

            # scaling numerik yang sama
            scaler_loaded = joblib.load("scaler_polusi.pkl")
            cont_loaded = joblib.load("continuous_columns_polusi.pkl")
            cont_loaded = [c for c in cont_loaded if c in base_row.columns]
            if len(cont_loaded) > 0:
                base_row[cont_loaded] = scaler_loaded.transform(base_row[cont_loaded])

            model_loaded = joblib.load("model_polusi.pkl")
            le_loaded = joblib.load("label_encoder_polusi.pkl")

            pred = model_loaded.predict(base_row)[0]
            proba = model_loaded.predict_proba(base_row)[0]
            label = le_loaded.inverse_transform([pred])[0]

            st.success(f"Hasil Prediksi Kategori: **{label}**")

            prob_df = pd.DataFrame({
                "Kategori": le_loaded.classes_,
                "Probabilitas": proba
            }).sort_values("Probabilitas", ascending=False)

            prob_chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X("Probabilitas:Q", title="Probabilitas"),
                y=alt.Y("Kategori:N", sort="-x"),
                tooltip=["Kategori", alt.Tooltip("Probabilitas:Q", format=".3f")]
            ).properties(height=220, title="Probabilitas Prediksi per Kategori")

            st.altair_chart(prob_chart, use_container_width=True)
