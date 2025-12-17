import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Polusi Udara Jakarta",
    page_icon="🌫️",
    layout="wide"
)

# =========================
# HELPERS
# =========================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )
    return df

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)

    # rename supaya konsisten dengan contoh umum
    rename_map = {
        "pm2.5": "pm25",
        "kategori": "categori",
        "category": "categori",
        "station": "location",
        "lokasi": "location",
        "wilayah": "location",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # wajib ada
    required = ["tanggal", "categori"]
    for r in required:
        if r not in df.columns:
            st.error(f"Kolom wajib '{r}' tidak ditemukan di dataset.")
            st.stop()

    # parse tanggal
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal"]).sort_values("tanggal")

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    polutan_cols = [c for c in ["pm10", "pm25", "so2", "co", "o3", "no2"] if c in df.columns]
    if polutan_cols:
        df["total_polutan"] = df[polutan_cols].sum(axis=1)
        df["mean_polutan"]  = df[polutan_cols].mean(axis=1)
        # dominansi (hindari div 0)
        df["dominance_ratio"] = np.where(df["total_polutan"] > 0, df["max"] / df["total_polutan"], np.nan)

        # z-score per polutan (untuk perbandingan skala)
        for c in polutan_cols:
            std = df[c].std(ddof=0)
            df[f"{c}_z"] = (df[c] - df[c].mean()) / (std if std != 0 else 1)

    # fitur waktu
    df["tahun"] = df["tanggal"].dt.year
    df["bulan"] = df["tanggal"].dt.month
    df["nama_bulan"] = df["tanggal"].dt.strftime("%b")
    df["hari"] = df["tanggal"].dt.day
    df["hari_dalam_minggu"] = df["tanggal"].dt.day_name()
    df["minggu_ke"] = df["tanggal"].dt.isocalendar().week.astype(int)

    # rolling (opsional)
    if "max" in df.columns:
        df["max_rolling_7d"] = df["max"].rolling(7, min_periods=1).mean()

    return df

# =========================
# MAIN DASHBOARD
# =========================
def chart():
    st.title("🌫️ Dashboard Visualisasi Polusi Udara Jakarta")
    st.caption("Interaktif: filter kategori/lokasi/tanggal, analisis tren polutan, korelasi, dan distribusi.")

    # ===== Load
    df = load_data("C:/Users/lenovo/Downloads/machine learning 2/polusijakarta.csv")
    df = add_features(df)

    # kolom polutan yang tersedia
    polutan_cols = [c for c in ["pm10", "pm25", "so2", "co", "o3", "no2"] if c in df.columns]
    location_col = "location" if "location" in df.columns else None
    critical_col = "critical" if "critical" in df.columns else None
    max_col = "max" if "max" in df.columns else None

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.header("🔎 Filter")
    min_date, max_date = df["tanggal"].min(), df["tanggal"].max()

    date_range = st.sidebar.date_input(
        "Rentang Tanggal",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    kategori_list = sorted(df["categori"].dropna().unique().tolist())
    selected_cat = st.sidebar.multiselect("Kategori", kategori_list, default=kategori_list)

    if location_col:
        loc_list = sorted(df[location_col].astype(str).dropna().unique().tolist())
        selected_loc = st.sidebar.multiselect("Lokasi", loc_list, default=loc_list)
    else:
        selected_loc = []

    if critical_col:
        crit_list = sorted(df[critical_col].astype(str).dropna().unique().tolist())
        selected_crit = st.sidebar.multiselect("Polutan Dominan (critical)", crit_list, default=crit_list)
    else:
        selected_crit = []

    # =========================
    # APPLY FILTER
    # =========================
    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["tanggal"] >= start) & (filtered["tanggal"] <= end)]

    if selected_cat:
        filtered = filtered[filtered["categori"].isin(selected_cat)]

    if location_col and selected_loc:
        filtered = filtered[filtered[location_col].astype(str).isin(selected_loc)]

    if critical_col and selected_crit:
        filtered = filtered[filtered[critical_col].astype(str).isin(selected_crit)]

    # =========================
    # METRICS
    # =========================
    st.subheader("📊 Ringkasan Utama")
    total = len(filtered)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Data", total)

    if max_col and total > 0:
        c2.metric("Max Tertinggi", int(filtered["max"].max()))
        c3.metric("Rata-rata Max", round(float(filtered["max"].mean()), 2))
        c4.metric("Median Max", round(float(filtered["max"].median()), 2))
    else:
        c2.metric("Max Tertinggi", "-")
        c3.metric("Rata-rata Max", "-")
        c4.metric("Median Max", "-")

    # kategori terbanyak
    if total > 0:
        top_cat = filtered["categori"].value_counts().idxmax()
        c5.metric("Kategori Dominan", str(top_cat))
    else:
        c5.metric("Kategori Dominan", "-")

    st.divider()

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("📄 Preview Data (setelah filter)")
    st.dataframe(filtered, use_container_width=True, height=260)

    st.divider()

    # =========================
    # 1) DISTRIBUSI KATEGORI
    # =========================
    st.subheader("📌 Distribusi Kategori Kualitas Udara")
    bar_cat = filtered["categori"].value_counts().reset_index()
    bar_cat.columns = ["Kategori", "Jumlah"]

    chart_cat = alt.Chart(bar_cat).mark_bar().encode(
        x=alt.X("Kategori:N", sort="-y"),
        y=alt.Y("Jumlah:Q"),
        color="Kategori:N",
        tooltip=["Kategori", "Jumlah"]
    ).properties(height=320)

    st.altair_chart(chart_cat, use_container_width=True)

    # =========================
    # 2) TREN HARIAN (MAX + ROLLING)
    # =========================
    if max_col and len(filtered) > 0:
        st.subheader("📈 Tren Harian Nilai Maksimum (max) + Rolling 7 Hari")

        base = alt.Chart(filtered).encode(
            x=alt.X("tanggal:T", title="Tanggal")
        )

        line_max = base.mark_line().encode(
            y=alt.Y("max:Q", title="Nilai")
        )

        line_roll = base.mark_line(strokeDash=[6, 4]).encode(
            y=alt.Y("max_rolling_7d:Q", title="Nilai")
        )

        st.altair_chart((line_max + line_roll).interactive().properties(height=320), use_container_width=True)

    # =========================
    # 3) MULTI-LINE POLUTAN
    # =========================
    if polutan_cols and len(filtered) > 0:
        st.subheader("🧪 Tren Polutan Harian (Multi-line)")

        selected_polutans = st.multiselect(
            "Pilih polutan untuk ditampilkan",
            polutan_cols,
            default=polutan_cols
        )

        if selected_polutans:
            long_df = filtered.melt(
                id_vars=["tanggal", "categori"] + ([location_col] if location_col else []),
                value_vars=selected_polutans,
                var_name="Polutan",
                value_name="Nilai"
            )

            line_pol = alt.Chart(long_df).mark_line().encode(
                x=alt.X("tanggal:T", title="Tanggal"),
                y=alt.Y("Nilai:Q", title="Konsentrasi"),
                color="Polutan:N",
                tooltip=["tanggal:T", "Polutan:N", "Nilai:Q"]
            ).interactive().properties(height=340)

            st.altair_chart(line_pol, use_container_width=True)

    st.divider()

    # =========================
    # 4) STACKED BAR: KATEGORI vs LOKASI
    # =========================
    if location_col and len(filtered) > 0:
        st.subheader("📍 Proporsi Kategori per Lokasi (Stacked Normalized)")

        grp = filtered.groupby([location_col, "categori"]).size().reset_index(name="Jumlah")

        stacked = alt.Chart(grp).mark_bar().encode(
            x=alt.X(f"{location_col}:N", title="Lokasi"),
            y=alt.Y("Jumlah:Q", stack="normalize", title="Proporsi"),
            color=alt.Color("categori:N", title="Kategori"),
            tooltip=[location_col, "categori", "Jumlah"]
        ).properties(height=340)

        st.altair_chart(stacked, use_container_width=True)

    # =========================
    # 5) BOXPLOT: POLUTAN vs KATEGORI
    # =========================
    if polutan_cols and len(filtered) > 0:
        st.subheader("📦 Sebaran Polutan per Kategori (Boxplot)")
        p = st.selectbox("Pilih Polutan", polutan_cols, index=polutan_cols.index("pm25") if "pm25" in polutan_cols else 0)

        box = alt.Chart(filtered).mark_boxplot(extent=1.5).encode(
            x=alt.X("categori:N", title="Kategori"),
            y=alt.Y(f"{p}:Q", title=p.upper()),
            color=alt.Color("categori:N", legend=None)
        ).properties(height=320)

        st.altair_chart(box, use_container_width=True)

    # =========================
    # 6) HEATMAP KORELASI POLUTAN
    # =========================
    if polutan_cols and len(filtered) > 2:
        st.subheader("🔥 Korelasi Antar Polutan (Heatmap)")

        corr = filtered[polutan_cols].corr().reset_index().melt("index")
        corr.columns = ["Polutan_X", "Polutan_Y", "Korelasi"]

        heatmap = alt.Chart(corr).mark_rect().encode(
            x=alt.X("Polutan_X:N", title=""),
            y=alt.Y("Polutan_Y:N", title=""),
            color=alt.Color("Korelasi:Q", scale=alt.Scale(scheme="redblue")),
            tooltip=["Polutan_X", "Polutan_Y", alt.Tooltip("Korelasi:Q", format=".2f")]
        ).properties(height=360)

        st.altair_chart(heatmap, use_container_width=True)

    st.divider()

    # =========================
    # 7) DOMINANT POLLUTANT INSIGHT
    # =========================
    if critical_col and len(filtered) > 0:
        st.subheader("🏆 Polutan Dominan (critical) Paling Sering Muncul")

        dom = filtered[critical_col].value_counts().reset_index()
        dom.columns = ["Polutan_Dominan", "Frekuensi"]

        dom_bar = alt.Chart(dom).mark_bar().encode(
            x=alt.X("Polutan_Dominan:N", sort="-y", title="Polutan Dominan"),
            y=alt.Y("Frekuensi:Q", title="Frekuensi"),
            tooltip=["Polutan_Dominan", "Frekuensi"]
        ).properties(height=300)

        st.altair_chart(dom_bar, use_container_width=True)

    # =========================
    # 8) FEATURE TURUNAN (TOTAL & DOMINANCE)
    # =========================
    if "total_polutan" in filtered.columns and len(filtered) > 0:
        st.subheader("🧠 Fitur Turunan: Total Polutan & Dominance Ratio")

        cA, cB = st.columns(2)

        with cA:
            line_total = alt.Chart(filtered).mark_line().encode(
                x=alt.X("tanggal:T", title="Tanggal"),
                y=alt.Y("total_polutan:Q", title="Total Polutan"),
                tooltip=["tanggal:T", "total_polutan:Q"]
            ).interactive().properties(height=280)
            st.altair_chart(line_total, use_container_width=True)

        with cB:
            if "dominance_ratio" in filtered.columns:
                line_dom = alt.Chart(filtered).mark_line().encode(
                    x=alt.X("tanggal:T", title="Tanggal"),
                    y=alt.Y("dominance_ratio:Q", title="Max / Total Polutan"),
                    tooltip=["tanggal:T", alt.Tooltip("dominance_ratio:Q", format=".3f")]
                ).interactive().properties(height=280)
                st.altair_chart(line_dom, use_container_width=True)

# Jalankan
if __name__ == "__main__":
    chart()
