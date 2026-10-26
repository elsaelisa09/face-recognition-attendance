from pathlib import Path

import streamlit as st
from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


ROOT = Path(__file__).resolve().parent


def main():
    try:
        config = dict(st.secrets["database_supabase"])
    except Exception:
        print(
            "Konfigurasi gagal dibaca. Periksa bagian "
            "[database_supabase] di .streamlit/secrets.toml."
        )
        return 1

    certificate = Path(config["sslrootcert"])

    if not certificate.is_absolute():
        certificate = ROOT / certificate

    if not certificate.is_file():
        print("Sertifikat tidak ditemukan:", certificate)
        return 1

    url = URL.create(
        drivername="postgresql+psycopg",
        username=config["username"],
        password=config["password"],
        host=config["host"],
        port=int(config["port"]),
        database=config["name"],
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 10,
            "sslmode": "verify-full",
            "sslrootcert": str(certificate.resolve()),
        },
    )

    print("Menghubungkan ke Supabase...", flush=True)

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SET LOCAL statement_timeout = '15s'")
            )
            connection.execute(
                text("SET LOCAL lock_timeout = '5s'")
            )

            info = connection.execute(
                text("""
                    SELECT
                        current_database() AS database_name,
                        current_user AS database_user
                """)
            ).mappings().one()

            if info["database_user"] != "face_attendance_app":
                raise RuntimeError(
                    "Koneksi tidak menggunakan akun aplikasi."
                )

            print("Database:", info["database_name"])
            print("Pengguna:", info["database_user"])

            # Nama tabel berasal dari daftar tetap,
            # bukan dari input pengguna.
            for table_name in (
                "persons",
                "face_references",
                "attendance",
            ):
                count = connection.execute(
                    text(
                        f"SELECT COUNT(*) "
                        f"FROM public.{table_name}"
                    )
                ).scalar_one()

                print(f"Jumlah {table_name}: {count}")

            # Uji izin penguncian untuk pendaftaran.
            connection.execute(
                text("""
                    LOCK TABLE public.persons
                    IN SHARE ROW EXCLUSIVE MODE
                """)
            )

            # Uji query penguncian yang dipakai saat absensi.
            connection.execute(
                text("""
                    SELECT person_id
                    FROM public.persons
                    LIMIT 1
                    FOR SHARE
                """)
            ).all()

            connection.rollback()

        print("\nKoneksi Supabase berhasil.")
        print("SSL: verify-full")
        print("Akses baca dan perintah penguncian berhasil.")
        return 0

    except SQLAlchemyError as exc:
        # Tampilkan pesan driver tanpa mencetak connection string.
        detail = str(getattr(exc, "orig", exc))
        password = str(config["password"])

        if password:
            detail = detail.replace(password, "***")

        print("\nPemeriksaan Supabase gagal:")
        print(detail)
        return 1

    except RuntimeError as exc:
        print("\nPemeriksaan gagal:", str(exc))
        return 1

    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())