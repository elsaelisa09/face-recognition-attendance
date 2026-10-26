import streamlit as st
from sqlalchemy import URL, create_engine, text


def main():
    config = st.secrets["database"]

    connection_url = URL.create(
        drivername="postgresql+psycopg",
        username=config["username"],
        password=config["password"],
        host=config["host"],
        port=int(config["port"]),
        database=config["name"],
    )

    engine = create_engine(
        connection_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT current_database() AS database_name, "
                    "current_user AS username"
                )
            ).mappings().one()

            print("Koneksi PostgreSQL berhasil.")
            print("Database:", result["database_name"])
            print("Pengguna:", result["username"])

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()