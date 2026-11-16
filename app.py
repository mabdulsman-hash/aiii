# app.py - Aplikasi Streamlit: Prediksi Risiko Gagal Bayar Debitur
# Jalankan: streamlit run app.py   (file model.joblib harus satu folder dengan file ini)

import joblib
import pandas as pd
import streamlit as st

LABELS = {
    "person_age": "Usia debitur (tahun)",
    "person_income": "Pendapatan tahunan",
    "person_emp_length": "Lama bekerja (tahun)",
    "loan_amnt": "Jumlah pinjaman",
    "loan_int_rate": "Suku bunga pinjaman (%)",
    "cb_person_cred_hist_length": "Lama riwayat kredit (tahun)",
    "person_home_ownership": "Status tempat tinggal",
    "loan_intent": "Tujuan pinjaman",
    "loan_grade": "Grade pinjaman",
    "cb_person_default_on_file": "Pernah gagal bayar sebelumnya",
}


def load_artifact(path="model.joblib"):
    return joblib.load(path)


def build_input_frame(values, features):
    """Susun satu baris data sesuai urutan kolom saat training."""
    row = dict(values)
    income = float(row.get("person_income", 0) or 0)
    row["loan_percent_income"] = (float(row["loan_amnt"]) / income) if income > 0 else 0.0
    return pd.DataFrame([row])[list(features)]


def predict_default(artifact, values):
    frame = build_input_frame(values, artifact["features"])
    proba = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    return proba, proba >= artifact["threshold"]


def main():
    st.set_page_config(page_title="Prediksi Gagal Bayar Debitur", page_icon="💳")
    st.title("💳 Prediksi Risiko Gagal Bayar Debitur")
    st.write(
        "Masukkan data debitur, lalu model machine learning akan memperkirakan "
        "probabilitas debitur gagal bayar."
    )

    try:
        artifact = st.cache_resource(load_artifact)()
    except FileNotFoundError:
        st.error("File model.joblib tidak ditemukan. Jalankan notebook sampai bagian "
                 "'Simpan Model', lalu letakkan model.joblib di folder yang sama dengan app.py.")
        st.stop()

    num_cols = [c for c in artifact["num_cols"] if c != "loan_percent_income"]
    cat_cols = artifact["cat_cols"]

    values = {}
    left, right = st.columns(2)
    for i, col in enumerate(num_cols):
        lo, med, hi = artifact["num_ranges"][col]
        box = left if i % 2 == 0 else right
        values[col] = box.number_input(
            LABELS.get(col, col), min_value=float(lo), max_value=float(hi),
            value=float(med),
        )
    for i, col in enumerate(cat_cols):
        box = left if i % 2 == 0 else right
        values[col] = box.selectbox(LABELS.get(col, col), artifact["categories"][col])

    if st.button("Prediksi", type="primary"):
        proba, is_default = predict_default(artifact, values)
        st.metric("Probabilitas gagal bayar", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
        if is_default:
            st.error("⚠️ Berisiko TINGGI gagal bayar (di atas ambang batas model).")
        else:
            st.success("✅ Risiko gagal bayar RENDAH (di bawah ambang batas model).")
        st.caption(f"Ambang batas keputusan model: {artifact['threshold']:.2f}")

    with st.expander("Tentang model"):
        st.write(f"Algoritma: **{artifact['model_name']}**")
        st.json(artifact["metrics"])
        st.caption(
            "Aplikasi ini alat bantu untuk tugas perkuliahan, bukan pengganti "
            "keputusan kredit yang sebenarnya."
        )


if __name__ == "__main__":
    main()
