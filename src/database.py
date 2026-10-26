from functools import lru_cache
from pathlib import Path

import streamlit as st
from sqlalchemy import URL, create_engine, event, text


@lru_cache(maxsize=1)
def get_engine():
    config = st.secrets["database_supabase"]

    project_root = Path(__file__).resolve().parents[1]
    certificate = Path(config["sslrootcert"])

    if not certificate.is_absolute():
        certificate = project_root / certificate

    if not certificate.is_file():
        raise FileNotFoundError(
            f"Sertifikat SSL tidak ditemukan: {certificate}"
        )

    engine = create_engine(
        URL.create(
            drivername="postgresql+psycopg",
            username=config["username"],
            password=config["password"],
            host=config["host"],
            port=int(config["port"]),
            database=config["name"],
        ),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        pool_timeout=30,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 10,
            "sslmode": "verify-full",
            "sslrootcert": str(certificate.resolve()),
        },
    )

    @event.listens_for(engine, "connect")
    def configure_connection(dbapi_connection, connection_record):
        previous_autocommit = dbapi_connection.autocommit

        try:
            # Simpan pengaturan sesi di luar transaksi aplikasi.
            dbapi_connection.autocommit = True

            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET extra_float_digits = 3")

        finally:
            dbapi_connection.autocommit = previous_autocommit

    return engine

def init_database():
    statements = [
        """
        CREATE TABLE IF NOT EXISTS public.persons (
            person_id VARCHAR(64) PRIMARY KEY,

            full_name VARCHAR(200) NOT NULL
                CHECK (length(trim(full_name)) > 0),

            centroid REAL[] NOT NULL
                CHECK (
                    array_ndims(centroid) = 1
                    AND cardinality(centroid) = 512
                    AND array_position(centroid, NULL) IS NULL
                ),

            is_active BOOLEAN NOT NULL DEFAULT TRUE,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS public.face_references (
            reference_id BIGSERIAL PRIMARY KEY,

            person_id VARCHAR(64) NOT NULL
                REFERENCES public.persons(person_id)
                ON DELETE CASCADE,

            reference_number SMALLINT NOT NULL
                CHECK (reference_number IN (1, 2)),

            embedding REAL[] NOT NULL
                CHECK (
                    array_ndims(embedding) = 1
                    AND cardinality(embedding) = 512
                    AND array_position(embedding, NULL) IS NULL
                ),

            source_image_id VARCHAR(128),

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT uq_person_reference
                UNIQUE (person_id, reference_number)
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS public.attendance (
            attendance_id BIGSERIAL PRIMARY KEY,

            person_id VARCHAR(64) NOT NULL
                REFERENCES public.persons(person_id)
                ON DELETE RESTRICT,

            checked_in_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            attendance_date DATE
                GENERATED ALWAYS AS (
                    (checked_in_at AT TIME ZONE 'Asia/Jakarta')::date
                ) STORED NOT NULL,

            similarity DOUBLE PRECISION NOT NULL,

            threshold_used DOUBLE PRECISION NOT NULL,

            CONSTRAINT uq_person_attendance_date
                UNIQUE (person_id, attendance_date)
        )
        """,
    ]

    engine = get_engine()

    # Seluruh pembuatan tabel dijalankan dalam satu transaksi.
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

    return engine


def main():
    engine = init_database()

    with engine.connect() as connection:
        table_names = connection.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN (
                      'persons',
                      'face_references',
                      'attendance'
                  )
                ORDER BY table_name
            """)
        ).scalars().all()

    print("Inisialisasi database berhasil.")
    print("Tabel tersedia:")

    for table_name in table_names:
        print(f"- {table_name}")

    print("Jumlah tabel:", len(table_names))


if __name__ == "__main__":
    main()