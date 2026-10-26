from pathlib import Path

import numpy as np
import streamlit as st
from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


ROOT = Path(__file__).resolve().parent

# Nama tabel dan kolom ditentukan oleh kode,
# bukan berasal dari input pengguna.
TABLES = {
    "persons": {
        "columns": (
            "person_id", "full_name", "centroid",
            "is_active", "created_at",
        ),
        "order": "person_id",
    },
    "face_references": {
        "columns": (
            "person_id", "reference_number", "embedding",
            "source_image_id", "created_at",
        ),
        "order": "person_id, reference_number",
    },
    "attendance": {
        "columns": (
            "person_id", "checked_in_at", "attendance_date",
            "similarity", "threshold_used",
        ),
        "order": "person_id, attendance_date",
    },
}

VECTOR_COLUMNS = {"centroid", "embedding"}


def build_engine(config, cloud=False):
    options = {"connect_timeout": 10}

    if cloud:
        certificate = Path(config["sslrootcert"])

        if not certificate.is_absolute():
            certificate = ROOT / certificate

        if not certificate.is_file():
            raise ValueError("Sertifikat Supabase tidak ditemukan.")

        options.update({
            "sslmode": "verify-full",
            "sslrootcert": str(certificate.resolve()),
        })

    return create_engine(
        URL.create(
            drivername="postgresql+psycopg",
            username=config["username"],
            password=config["password"],
            host=config["host"],
            port=int(config["port"]),
            database=config["name"],
        ),
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        hide_parameters=True,
        connect_args=options,
    )


def read_rows(connection, table):
    if table == "persons":
        previous_precision = connection.execute(
            text("SHOW extra_float_digits")
        ).scalar_one()

        print(
            "Pengaturan presisi sebelum membaca persons:",
            previous_precision,
            flush=True,
        )

    # Berlaku selama transaksi ini.
    # Mempertahankan presisi keluaran floating-point.
    connection.execute(
        text("SET LOCAL extra_float_digits = 3")
    )

    spec = TABLES[table]
    columns = ", ".join(spec["columns"])

    rows = connection.execute(
        text(
            f"SELECT {columns} FROM public.{table} "
            f"ORDER BY {spec['order']}"
        )
    ).mappings().all()

    return [dict(row) for row in rows]


def verify_rows(table, source_rows, target_rows):
    if len(source_rows) != len(target_rows):
        raise ValueError(f"Jumlah data berbeda: {table}")

    tolerance = 1e-7
    largest_difference = 0.0

    for source, target in zip(source_rows, target_rows):
        for column in TABLES[table]["columns"]:
            if column in VECTOR_COLUMNS:
                source_vector = np.asarray(
                    source[column],
                    dtype=np.float64,
                )

                target_vector = np.asarray(
                    target[column],
                    dtype=np.float64,
                )

                if (
                    source_vector.shape != (512,)
                    or target_vector.shape != (512,)
                ):
                    raise ValueError(
                        f"Ukuran vektor tidak sesuai: "
                        f"{table}.{column}"
                    )

                if (
                    not np.isfinite(source_vector).all()
                    or not np.isfinite(target_vector).all()
                ):
                    raise ValueError(
                        f"Ada angka tidak valid: {table}.{column}"
                    )

                difference = float(
                    np.max(
                        np.abs(source_vector - target_vector)
                    )
                )

                largest_difference = max(
                    largest_difference,
                    difference,
                )

                if not np.allclose(
                    source_vector,
                    target_vector,
                    rtol=0.0,
                    atol=tolerance,
                ):
                    raise ValueError(
                        f"Isi vektor berbeda: {table}.{column}. "
                        f"Person ID: {source['person_id']}. "
                        f"Selisih maksimum: {difference:.10e}. "
                        f"Batas: {tolerance:.10e}"
                    )

            elif source[column] != target[column]:
                raise ValueError(
                    f"Isi data berbeda: {table}.{column}. "
                    f"Person ID: {source['person_id']}"
                )

    if any(
        column in VECTOR_COLUMNS
        for column in TABLES[table]["columns"]
    ):
        print(
            f"Selisih maksimum vektor {table}: "
            f"{largest_difference:.10e}",
            flush=True,
        )

def main():
    try:
        local_config = dict(st.secrets["database"])
        cloud_config = dict(st.secrets["database_supabase"])
    except Exception:
        print(
            "Gagal membaca konfigurasi. Periksa bagian "
            "[database] dan [database_supabase] di secrets.toml."
        )
        return 1

    engines = []

    try:
        local_engine = build_engine(local_config)
        engines.append(local_engine)

        cloud_engine = build_engine(cloud_config, cloud=True)
        engines.append(cloud_engine)

        # Ambil snapshot yang konsisten dari database lokal.
        snapshot_engine = local_engine.execution_options(
            isolation_level="REPEATABLE READ"
        )

        print("Membaca database lokal...", flush=True)

        with snapshot_engine.connect() as source:
            with source.begin():
                source.execute(
                    text("SET TRANSACTION READ ONLY")
                )

                source_data = {
                    table: read_rows(source, table)
                    for table in TABLES
                }

        if not source_data["persons"]:
            raise ValueError(
                "Database lokal tidak memiliki identitas. "
                "Periksa konfigurasi [database]."
            )

        for table, rows in source_data.items():
            print(f"Data lokal {table}: {len(rows)}")

        print("\nMenyalin ke Supabase...", flush=True)

        # Seluruh INSERT dan verifikasi berada dalam satu transaksi.
        with cloud_engine.begin() as target:
            target.execute(
                text("SET LOCAL lock_timeout = '10s'")
            )
            target.execute(
                text("SET LOCAL statement_timeout = '120s'")
            )

            role = target.execute(
                text("SELECT current_user")
            ).scalar_one()

            if role != "face_attendance_app":
                raise ValueError(
                    "Tujuan tidak memakai akun face_attendance_app."
                )

            target.execute(text("""
                LOCK TABLE public.persons
                IN SHARE ROW EXCLUSIVE MODE
            """))

            # Hindari menimpa atau menggandakan data yang sudah ada.
            for table in TABLES:
                count = target.execute(
                    text(f"SELECT COUNT(*) FROM public.{table}")
                ).scalar_one()

                if count:
                    raise ValueError(
                        f"Tabel tujuan {table} sudah berisi data. "
                        "Migrasi dibatalkan."
                    )

            # Urutan: persons -> face_references -> attendance.
            for table, spec in TABLES.items():
                rows = source_data[table]

                if not rows:
                    continue

                # attendance_date dihitung otomatis oleh PostgreSQL.
                columns = [
                    column
                    for column in spec["columns"]
                    if column != "attendance_date"
                ]

                placeholders = [
                    (
                        f"CAST(:{column} AS REAL[])"
                        if column in VECTOR_COLUMNS
                        else f":{column}"
                    )
                    for column in columns
                ]

                statement = text(
                    f"INSERT INTO public.{table} "
                    f"({', '.join(columns)}) "
                    f"VALUES ({', '.join(placeholders)})"
                )

                parameters = [
                    {column: row[column] for column in columns}
                    for row in rows
                ]

                target.execute(statement, parameters)

            # Bandingkan seluruh kolom yang disalin,
            # termasuk vektor wajah dan tanggal absensi.
            for table in TABLES:
                copied_rows = read_rows(target, table)

                verify_rows(
                    table,
                    source_data[table],
                    copied_rows,
                )

                print(
                    f"Terverifikasi {table}: {len(copied_rows)}",
                    flush=True,
                )

        # Pesan ini muncul setelah transaksi berhasil di-commit.
        print("\nMigrasi berhasil.")
        print("Seluruh data salinan telah diverifikasi.")
        print("Data lokal tetap tersedia.")
        return 0

    except SQLAlchemyError as exc:
        original = getattr(exc, "orig", exc)
        diagnostic = getattr(original, "diag", None)

        message = (
            getattr(diagnostic, "message_primary", None)
            or str(original)
        )

        for config in (local_config, cloud_config):
            password = str(config.get("password", ""))
            if password:
                message = message.replace(password, "***")

        print("\nMigrasi gagal:", message)
        print("Periksa data tujuan sebelum mencoba ulang.")
        return 1

    except ValueError as exc:
        print("\nMigrasi dibatalkan:", str(exc))
        return 1

    finally:
        for engine in engines:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())