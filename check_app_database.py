from sqlalchemy import text

from src.database import get_engine


def main():
    engine = get_engine()

    with engine.connect() as connection:
        user = connection.execute(
            text("SELECT current_user")
        ).scalar_one()

        precision = connection.execute(
            text("SHOW extra_float_digits")
        ).scalar_one()

        print("Pengguna database:", user)
        print("Presisi angka    :", precision)

        for table in ("persons", "face_references", "attendance"):
            count = connection.execute(
                text(f"SELECT COUNT(*) FROM public.{table}")
            ).scalar_one()

            print(f"{table}: {count}")

    print("\nKoneksi database aplikasi berhasil.")


if __name__ == "__main__":
    main()