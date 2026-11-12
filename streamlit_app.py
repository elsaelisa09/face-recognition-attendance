import base64
import hashlib
import html
import io
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import streamlit as st

from src.face_pipeline import FaceRecognitionPipeline, draw_faces
from src.registration_page import render_registration
from src.attendance_service import mark_attendance

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Sistem Kehadiran",
    page_icon="📋",
    layout="wide",
)

STYLE = """
<style>
:root {
    --fr-panel-height: clamp(465px, calc(100dvh - 165px), 590px);
    --fr-preview-height: clamp(150px, calc(100dvh - 465px), 290px);
}

.block-container {
    max-width: 1200px;
    padding: 1.25rem 2rem 1rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.st-key-fr_input,
.st-key-fr_output {
    min-height: var(--fr-panel-height);
    border-radius: 16px;
    background: #ffffff;
    box-sizing: border-box;
}

[data-testid="stFileUploaderDropzone"] {
    padding: .65rem;
}

/* Ukuran teks petunjuk pada kotak unggah foto */
[data-testid="stFileUploaderDropzone"] span {
    font-size: 13px !important;
}

[data-testid="stFileUploaderDropzone"] small {
    display: block;
    font-size: 11px !important;
}

[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    max-height: 300px;
    object-fit: contain;
}

.fr-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding-bottom: .35rem;
}

.fr-header h1 {
    font-size: 1.65rem;
    line-height: 1.3;
    letter-spacing: -.04em;
    margin: 0;
    padding: 0;
    color: #14213d;
}

.fr-header p {
    color: #64748b;
    font-size: .87rem;
    margin: .3rem 0 0;
}

.fr-badge {
    background: #e8efff;
    color: #1d4ed8;
    border-radius: 99px;
    padding: .45rem .8rem;
    font-size: .78rem;
    white-space: nowrap;
}

.fr-title {
    font-size: 1rem;
    font-weight: 650;
    color: #14213d;
    margin: 0;
}

.fr-preview {
    height: var(--fr-preview-height);
    width: 100%;
    box-sizing: border-box;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    margin: 0 0 .65rem;
}

.fr-preview img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.fr-placeholder {
    text-align: center;
    color: #64748b;
    padding: 1rem;
    font-size: .85rem;
}

.fr-placeholder strong {
    display: block;
    color: #334155;
    margin-bottom: .4rem;
}

.fr-result {
    min-height: calc(var(--fr-panel-height) - 75px);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: .85rem;
    padding: 1rem;
}

.fr-result h2 {
    font-size: 1.3rem;
    line-height: 1.4;
    margin: 0;
    padding: 0;
    color: #14213d;
    overflow-wrap: anywhere;
}

.fr-result p {
    font-size: .82rem;
    color: #64748b;
    margin: 0;
}

.fr-symbol {
    font-size: 2.5rem;
    color: #94a3b8;
    line-height: 1;
    margin-bottom: .4rem;
}

.fr-crop {
    width: 112px;
    height: 112px;
    object-fit: contain;
    border-radius: 14px;
    border: 3px solid #ffffff;
    box-shadow: 0 4px 18px #14213d14;
}

.fr-status {
    font-size: .75rem;
    font-weight: 650;
    border-radius: 99px;
    padding: .3rem .75rem;
    background: #dcfce7;
    color: #166534;
}

.fr-unknown {
    background: #fff1d6;
    color: #92400e;
}

.fr-scores {
    display: flex;
    width: 100%;
    max-width: 350px;
    gap: .75rem;
}

.fr-score {
    flex: 1;
    background: #f1f5f9;
    border-radius: 12px;
    padding: .7rem .5rem;
}

.fr-score span {
    display: block;
    color: #64748b;
    font-size: .73rem;
}

.fr-score strong {
    display: block;
    font-size: 1.3rem;
    color: #14213d;
    margin-top: .2rem;
}

.fr-note {
    color: #7c6520;
    font-size: .75rem;
    line-height: 1.5;
    margin: 0;
}

@media (max-height: 740px) and (min-width: 701px) {
    .block-container {
        padding-top: .7rem;
    }

    .st-key-fr_input [data-testid="stVerticalBlock"] {
        gap: .5rem;
    }

    .fr-result {
        gap: .65rem;
        padding: .7rem;
    }
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem;
    }

    .fr-header h1 {
        font-size: 1.3rem;
    }

    .fr-header p {
        font-size: .77rem;
    }

    .fr-badge {
        font-size: .7rem;
        padding: .4rem .6rem;
    }

    .st-key-fr_input,
    .st-key-fr_output {
        min-height: 0;
    }

    .fr-preview {
        height: 240px;
    }

    .fr-result {
        min-height: 300px;
    }
}

/* Attendance page visual system */
.stApp { background: #f5f7fb; color: #14213d; }
.block-container { max-width: 1320px; padding-top: 1.2rem; }
.fr-header { padding: 1.1rem 0 1rem; }
.fr-eyebrow { color: #2563eb; font-size: .7rem; font-weight: 700; letter-spacing: .12em; margin-bottom: .45rem; }
.fr-header h1 { font-size: 2rem; letter-spacing: -.04em; }
.fr-header p { font-size: .92rem; margin-top: .45rem; }
.fr-badge { background: #fff; color: #334155; border: 1px solid #e2e8f0; font-size: .8rem; font-weight: 600; }
.st-key-fr_input, .st-key-fr_output { border: 1px solid #e2e8f0 !important; box-shadow: 0 5px 20px rgba(15,23,42,.035); border-radius: 18px; }
.fr-title { font-size: 1.08rem; font-weight: 700; margin-bottom: .35rem; }
.fr-preview { border: 1px dashed #cbd5e1; background: #f7f9fc; border-radius: 14px; }
.fr-placeholder strong { color: #25344d; font-size: .96rem; }
.fr-result h2 { font-size: 1.45rem; }
.fr-crop { width: 104px; height: 104px; border-radius: 50%; object-fit: cover; }
.fr-scores { max-width: 340px; }
.fr-score { background: #f6f8fc; }
.fr-note { color: #64748b; padding: .2rem 0; }
.fr-guide {
    color: #334155;
    background: #eef4ff;
    border: 1px solid #dbeafe;
    border-radius: 12px;
    padding: .7rem .9rem;
    margin: .15rem 0 .8rem;
    font-size: .84rem;
    line-height: 1.5;
}
.st-key-fr_input [data-testid="stBaseButton-primary"] { border-radius: 10px; }
/* Compact layout for typical laptop viewports. The page remains scrollable
   when browser zoom, a short screen, or additional content requires it. */
@media (min-width: 701px) {
    :root {
        --fr-panel-height: clamp(345px, calc(100dvh - 315px), 475px);
        --fr-preview-height: clamp(90px, calc(100dvh - 565px), 195px);
    }

    .block-container { padding: .55rem 1.5rem .6rem; }
    .block-container > [data-testid="stVerticalBlock"] { gap: .4rem; }
    .fr-header { padding: .25rem 0 .2rem; }
    .fr-header h1 { font-size: 1.55rem; }
    .fr-header p { font-size: .83rem; margin-top: .2rem; }
    .fr-eyebrow { margin-bottom: .15rem; }
    .fr-guide { padding: .45rem .7rem; margin: 0 0 .15rem; font-size: .8rem; }
    .st-key-fr_input [data-testid="stVerticalBlock"],
    .st-key-fr_output [data-testid="stVerticalBlock"] { gap: .35rem; }
    [data-testid="stFileUploaderDropzone"] { padding: .4rem; }
    .fr-title { font-size: 1rem; margin-bottom: 0; }
    .fr-preview { margin-bottom: .2rem; }
    .fr-result { min-height: calc(var(--fr-panel-height) - 90px); gap: .4rem; padding: .4rem; }
    .fr-result h2 { font-size: 1.2rem; }
    .fr-crop { width: 78px; height: 78px; }
    .fr-score { padding: .4rem .5rem; }
    .fr-score strong { font-size: 1.05rem; }
    .fr-note { font-size: .7rem; line-height: 1.35; }
}
@media (max-width: 700px) {
    .block-container { padding: .75rem .9rem 1.2rem; }
    .fr-header { align-items: flex-start; flex-direction: column; gap: .6rem; padding: .45rem 0; }
    .fr-header h1 { font-size: 1.6rem; }
    .fr-guide { margin-bottom: .55rem; padding: .65rem .75rem; }
    .st-key-fr_input, .st-key-fr_output { min-height: 0; }
    .fr-preview { height: clamp(170px, 30dvh, 240px); }
    .fr-result { min-height: 220px; padding: .8rem; }
    .fr-scores { max-width: 100%; }
}

/* Keep the introduction in normal flow so text never overlaps. */
.fr-intro { display: flex; flex-direction: column; gap: .7rem; }
.fr-header { padding: 0; margin: 0; min-height: 0; }
.fr-heading { min-width: 0; }
.fr-header h1 { display: block; line-height: 1.2; margin: 0; padding: 0; }
.fr-subtitle { color: #64748b; font-size: .84rem; line-height: 1.45; margin-top: .3rem; }
.fr-guide { margin: 0; line-height: 1.45; }

@media (min-width: 701px) {
    :root {
        --fr-panel-height: clamp(360px, calc(100dvh - 350px), 450px);
        --fr-preview-height: clamp(85px, calc(100dvh - 590px), 160px);
    }
    .block-container { max-width: 1200px; padding: .7rem 2rem 1rem; }
    .fr-intro { gap: .65rem; }
    .fr-header { padding: 0; }
    .fr-guide { padding: .55rem .8rem; }
    /* Stretch the two cards together, while allowing their contents to grow. */
    div[data-testid="stHorizontalBlock"]:has(.st-key-fr_input) {
        align-items: stretch;
    }
    div[data-testid="stHorizontalBlock"]:has(.st-key-fr_input) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
    }
    .st-key-fr_input, .st-key-fr_output { height: 100%; }
    .fr-result { min-height: calc(var(--fr-panel-height) - 85px); }
    .fr-note { margin: 0; }
}
@media (max-width: 700px) {
    .fr-intro { gap: .65rem; }
    .fr-header { flex-direction: row; align-items: flex-start; flex-wrap: wrap; gap: .5rem; }
    .fr-header h1 { font-size: 1.5rem; }
    .fr-subtitle { font-size: .8rem; }
    .fr-guide { margin: 0; }
    .st-key-fr_input, .st-key-fr_output { height: auto; min-height: 0; }
}

/* Keterangan tim dibuat lebih ringkas pada desktop dan HP. */
p.fr-note {
    font-size: 14px !important;
    line-height: 1.4;
}
/* Jarak antarbagian halaman kehadiran. */
.fr-intro {
    gap: 1rem;
    padding-bottom: 1rem;
}

.fr-guide {
    padding: .7rem .9rem;
}

p.fr-note {
    padding-top: .7rem;
}

@media (min-width: 701px) {
    .fr-intro { gap: 1rem; }
    .fr-guide { padding: .7rem .9rem; }
    .st-key-fr_input [data-testid="stVerticalBlock"],
    .st-key-fr_output [data-testid="stVerticalBlock"] {
        gap: .65rem;
    }
}

@media (max-width: 700px) {
    .fr-intro { gap: .85rem; padding-bottom: .8rem; }
    .st-key-fr_input [data-testid="stVerticalBlock"],
    .st-key-fr_output [data-testid="stVerticalBlock"] {
        gap: .75rem;
    }
}

</style>
"""


@st.cache_resource(show_spinner=False)
def load_pipeline(config_signature):
    return FaceRecognitionPipeline(ROOT)


def image_uri(image):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=88,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def show_preview(image=None):
    if image is None:
        content = (
            '<div class="fr-placeholder">'
            "<strong>Foto akan tampil di sini</strong>"
            "Unggah foto atau gunakan kamera."
            "</div>"
        )
    else:
        content = (
            f'<img src="{image_uri(image)}" '
            'alt="Foto masukan dengan nomor wajah">'
        )

    st.markdown(
        f'<div class="fr-preview">{content}</div>',
        unsafe_allow_html=True,
    )


def show_result(result, message):
    if result is None:
        st.markdown(
            '<div class="fr-result">'
            '<div class="fr-symbol">◎</div>'
            "<h2>Siap melakukan absensi</h2>"
            f"<p>{html.escape(message)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return


    attendance_status = result.get("attendance_status")

    if attendance_status == "recorded":
        st.success(
            "Absensi berhasil disimpan: "
            + result["checked_in_at_wib"]
        )

    elif attendance_status == "already_recorded":
        st.info(
            "Sudah absen hari ini. Waktu pertama: "
            + result["checked_in_at_wib"]
        )

    if "gallery_size" in result:
        st.caption(
            f"Identitas aktif yang dibandingkan: "
            f"{result['gallery_size']}"
        )

    accepted = result["accepted"]

    name = (
        html.escape(str(result["person_name"]))
        if accepted
        else "Tidak dikenal"
    )

    status = (
        "Identitas dikenali"
        if accepted
        else "Wajah belum dikenali"
    )

    status_class = (
        "fr-status"
        if accepted
        else "fr-status fr-unknown"
    )

    st.markdown(
        '<div class="fr-result">'
        f'<img class="fr-crop" '
        f'src="{image_uri(result["aligned_face"])}" '
        'alt="Wajah yang diproses">'
        f'<span class="{status_class}">{status}</span>'
        f"<h2>{name}</h2>"
        '<div class="fr-scores">'
        '<div class="fr-score">'
        "<span>Skor kemiripan</span>"
        f'<strong>{result["similarity"]:.4f}</strong>'
        "</div>"
        '<div class="fr-score">'
        "<span>Ambang penerimaan</span>"
        f'<strong>{result["threshold"]:.4f}</strong>'
        "</div>"
        "</div>"
        "<p>Skor kemiripan bukan probabilitas identitas.</p>"
        f'<p>Alignment &amp; pengenalan: '
        f'{result["recognition_ms"]:.0f} ms</p>'
        "</div>",
        unsafe_allow_html=True,
    )


@st.dialog("Kehadiran berhasil dicatat 🎉")
def show_attendance_success(result):
    st.success("Absensi kamu sudah tercatat. Terima kasih!")
    st.text(f"Nama: {result['person_name']}")
    st.text(f"Waktu: {result['checked_in_at_wib']}")

    if st.button("Tutup", key="close_attendance_success"):
        st.rerun()


def clear_result():
    st.session_state.pop("face_result", None)
    st.session_state.pop("face_result_key", None)


def main():
    st.markdown(STYLE, unsafe_allow_html=True)

    try:
        config_path = (
            ROOT / "artifacts" / "inference_config.json"
        )

        signature = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()

        with st.spinner("Menyiapkan model..."):
            pipeline = load_pipeline(signature)

    except Exception:
        logging.exception("Failed to initialize pipeline")

        st.error(
            "Model atau konfigurasi belum siap. "
            "Lihat pesan pada terminal."
        )
        st.stop()

    menu = st.radio(
        "Menu",
        ["Absensi", "Daftar wajah"],
        horizontal=True,
        key="main_navigation",
    )

    if menu == "Daftar wajah":
        render_registration(pipeline)
        return

    today = datetime.now(timezone(timedelta(hours=7)))
    date_wib = today.strftime("%d/%m/%Y")

    st.markdown(
        '<div class="fr-intro">'
        '<div class="fr-header">'
        '<div class="fr-heading"><div class="fr-eyebrow">SISTEM KEHADIRAN</div>'
        '<h1>Sistem kehadiran</h1>'
        '</div>'
        f'<span class="fr-badge">📅 {date_wib} · WIB</span>'
        '</div>'
        '<div class="fr-guide">'
        '<strong>Sudah daftar?</strong> Langsung lakukan absensi. '
        '<strong>Belum?</strong> Pilih Daftar wajah dan siapkan 2 foto berbeda; '
        'setiap foto harus menampilkan wajah kamu sendiri.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1, 1],
        gap="medium",
    )

    result = None
    message = "Ambil atau unggah foto untuk memulai absensi."

    with left:
        with st.container(
            border=True,
            key="fr_input",
        ):
            st.markdown(
                '<p class="fr-title">Verifikasi wajah</p>',
                unsafe_allow_html=True,
            )

            source_type = st.radio(
                "Sumber foto",
                ["Unggah foto", "Kamera"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if source_type == "Unggah foto":
                uploaded = st.file_uploader(
                    "Pilih foto",
                    type=["jpg", "jpeg", "png", "webp", "mpo"],
                    label_visibility="collapsed",
                )
            else:
                with st.popover(
                    "Buka kamera / ambil ulang",
                    use_container_width=True,
                ):
                    uploaded = st.camera_input(
                        "Ambil foto wajah",
                        label_visibility="collapsed",
                    )

                    st.caption(
                        "Setelah mengambil foto, klik di luar "
                        "panel untuk melihat hasil."
                    )

            image = None
            faces = []
            upload_key = None
            error = None

            if uploaded is None:
                clear_result()

                for key in (
                    "face_upload_key",
                    "face_input_image",
                    "face_detections",
                ):
                    st.session_state.pop(key, None)

            else:
                data = uploaded.getvalue()

                upload_key = hashlib.sha256(
                    data + signature.encode()
                ).hexdigest()

                if (
                    st.session_state.get("face_upload_key")
                    != upload_key
                ):
                    clear_result()

                    try:
                        with st.spinner("Mendeteksi wajah..."):
                            image = pipeline.read_image(
                                io.BytesIO(data)
                            )
                            faces = pipeline.detect_faces(image)

                        st.session_state.update({
                            "face_upload_key": upload_key,
                            "face_input_image": image,
                            "face_detections": faces,
                        })

                    except Exception:
                        logging.exception(
                            "Failed to read or detect image"
                        )

                        image = None
                        faces = []
                        error = (
                            "Foto belum dapat diproses. "
                            "Coba foto lain."
                        )

                else:
                    image = st.session_state["face_input_image"]
                    faces = st.session_state["face_detections"]

            choice_key = f"face_choice_{upload_key}"

            valid_indices = [
                face["face_index"]
                for face in faces
            ]

            if len(faces) == 1:
                selected_index = valid_indices[0]
            else:
                selected_index = st.session_state.get(
                    choice_key
                )

            if selected_index not in valid_indices:
                selected_index = None

            # Tampilkan foto terlebih dahulu.
            # Tidak menggunakan st.empty() / placeholder bertingkat.
            if image is not None:
                preview = draw_faces(
                    image,
                    faces,
                    selected_index,
                )
            else:
                preview = None

            show_preview(preview)

            # Pilihan wajah dan tombol berada setelah foto.
            choice_col, button_col = st.columns(
                [1.3, 1],
                gap="small",
            )

            with choice_col:
                if len(faces) > 1:
                    selected_index = st.selectbox(
                        "Wajah yang dipilih",
                        options=[None] + valid_indices,
                        format_func=lambda value: (
                            "Pilih nomor wajah"
                            if value is None
                            else f"face {value}"
                        ),
                        key=choice_key,
                        label_visibility="collapsed",
                    )

                    message = (
                        "Pilih nomor pada foto, "
                        "lalu klik Catat kehadiran."
                    )

                elif faces:
                    st.caption(
                        "1 wajah terdeteksi · siap diproses"
                    )
                    message = (
                        "Klik Catat kehadiran untuk melanjutkan."
                    )

                elif image is not None:
                    st.caption("Tidak ada wajah terdeteksi")
                    message = (
                        "Coba foto dengan wajah "
                        "lebih jelas dan terang."
                    )

                else:
                    st.caption("JPG · PNG · WEBP · MPO")

            result_key = (
                upload_key,
                selected_index,
            )

            if (
                st.session_state.get("face_result_key")
                != result_key
            ):
                clear_result()

            with button_col:
                clicked = st.button(
                    "Catat kehadiran",
                    type="primary",
                    use_container_width=True,
                    disabled=selected_index is None,
                )

            if clicked:
                chosen = next(
                    face
                    for face in faces
                    if face["face_index"] == selected_index
                )

                try:
                    with st.spinner("Mengenali wajah..."):
                        recognition = mark_attendance(
                            image,
                            pipeline,
                            selected_face=chosen,
                        )

                    st.session_state["face_result"] = recognition
                    st.session_state["face_result_key"] = result_key

                except Exception:
                    logging.exception(
                        "Failed to recognize selected face"
                    )
                    clear_result()

                    error = (
                        "Wajah belum dapat diproses. "
                        "Coba foto lain."
                    )
                else:
                    if recognition.get("attendance_status") == "recorded":
                        show_attendance_success(recognition)

            if error:
                st.error(error)
                message = error

            if len(faces) > 1:
                st.caption(
                    "Pilih wajah utama sesuai label pada foto."
                )

            st.caption(
                "Foto kamu tidak disimpan. Saat daftar, sistem menyimpan "
                "nama dan fitur wajahmu dalam bentuk angka."
            )

            if (
                st.session_state.get("face_result_key")
                == result_key
            ):
                result = st.session_state.get("face_result")

    with right:
        with st.container(
            border=True,
            key="fr_output",
        ):
            st.markdown(
                '<p class="fr-title">Status kehadiran</p>',
                unsafe_allow_html=True,
            )

            show_result(result, message)

    st.markdown(
        '<p class="fr-note">'
        'Proyek mata kuliah Deep Learning · Tim Mata Trio Kwek Kwek<br>'
        'Elsa Elisa Yohana · Machzaul Harmansyah · Sikah Nubuahtul'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
