import logging

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.attendance_service import (
    DuplicateFaceError,
    RegistrationError,
    register_person,
)


def render_registration(pipeline):
    st.subheader("Daftar Wajah")

    st.caption(
        "Isi nama lengkap dan unggah dua foto dirimu "
        "dari pengambilan berbeda."
    )

    left, right = st.columns(
        [1, 1],
        gap="medium",
    )

    with left:
        with st.form(
            "face_registration_form",
            clear_on_submit=False,
        ):
            full_name = st.text_input(
                "Nama lengkap",
                max_chars=200,
                placeholder="Masukkan nama lengkap",
                key="registration_full_name",
            )

            photos = st.file_uploader(
                "Unggah tepat 2 foto",
                type=["jpg", "jpeg", "png", "webp", "mpo"],
                accept_multiple_files=True,
                key="registration_photos",
            )

            st.caption(
                "Setiap foto harus berisi tepat satu wajah."
            )

            submitted = st.form_submit_button(
                "Daftarkan wajah",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            st.session_state.pop(
                "registration_display_result",
                None,
            )

            try:
                with st.spinner(
                    "Memeriksa wajah dan menyimpan pendaftaran..."
                ):
                    result = register_person(
                        full_name=full_name,
                        photo_sources=photos,
                        pipeline=pipeline,
                    )

                st.session_state[
                    "registration_display_result"
                ] = result

            except DuplicateFaceError as exc:
                st.warning(str(exc))

                st.text(
                    f"Nama yang sudah terdaftar: {exc.full_name}"
                )

                st.caption(
                    f"Skor kemiripan: {exc.similarity:.4f} · "
                    f"Ambang pemeriksaan: {exc.threshold:.4f}"
                )

            except RegistrationError as exc:
                st.error(str(exc))

            except SQLAlchemyError:
                logging.exception(
                    "Database error during face registration"
                )

                st.error(
                    "Pendaftaran gagal disimpan. "
                    "Periksa koneksi database dan terminal."
                )

            except Exception:
                logging.exception(
                    "Unexpected face registration error"
                )

                st.error(
                    "Terjadi kesalahan. "
                    "Periksa pesan pada terminal."
                )

    with right:
        with st.container(border=True):
            st.markdown("**Hasil pendaftaran**")

            result = st.session_state.get(
                "registration_display_result"
            )

            if result is None:
                st.info(
                    "Hasil pendaftaran yang berhasil "
                    "akan tampil di sini."
                )

            else:
                st.success(
                    "Pendaftaran berhasil. Identitas sudah aktif."
                )

                st.caption("Nama lengkap")
                st.text(result["full_name"])

                st.caption("ID identitas")
                st.code(
                    result["person_id"],
                    language=None,
                )

                first, second = st.columns(2)

                with first:
                    st.image(
                        result["aligned_faces"][0],
                        caption="Referensi 1",
                        width=112,
                    )

                with second:
                    st.image(
                        result["aligned_faces"][1],
                        caption="Referensi 2",
                        width=112,
                    )