import hashlib
import json
from pathlib import Path

import numpy as np
from sqlalchemy import text

from src.database import get_engine


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"


def load_initial_gallery():
    gallery_path = ARTIFACTS_DIR / "gallery_70.npz"
    embeddings_path = ARTIFACTS_DIR / "arcface_embeddings.npz"
    config_path = ARTIFACTS_DIR / "inference_config.json"

    for path in (gallery_path, embeddings_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    # Periksa apakah artefak masih sesuai konfigurasi.
    for path, hash_key in [
        (gallery_path, "gallery_sha256"),
        (embeddings_path, "source_embeddings_sha256"),
    ]:
        actual_hash = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        if actual_hash != config[hash_key]:
            raise ValueError(
                f"Hash tidak sesuai konfigurasi: {path.name}"
            )

    with np.load(gallery_path, allow_pickle=False) as saved:
        centroids = saved["centroids"].astype(np.float32)
        person_ids = saved["person_ids"].astype(str)
        person_names = saved["person_names"].astype(str)
        reference_ids = saved["reference_image_ids"].astype(str)

    with np.load(embeddings_path, allow_pickle=False) as saved:
        embeddings = saved["embeddings"].astype(np.float32)
        image_ids = saved["image_ids"].astype(str)
        embedding_person_ids = saved["person_ids"].astype(str)

    # Validasi struktur galeri awal.
    if (
        centroids.shape != (70, 512)
        or person_ids.shape != (70,)
        or person_names.shape != (70,)
        or reference_ids.shape != (70, 2)
    ):
        raise ValueError(
            "Struktur galeri awal tidak sesuai. "
            "Diperlukan 70 identitas, centroid 512 dimensi, "
            "dan 2 referensi per identitas."
        )

    if len(set(person_ids)) != 70:
        raise ValueError("Ada person_id berulang.")

    if (
        image_ids.ndim != 1
        or embeddings.shape != (len(image_ids), 512)
        or embedding_person_ids.shape != image_ids.shape
        or len(set(image_ids)) != len(image_ids)
    ):
        raise ValueError(
            "Struktur file embedding tidak sesuai."
        )

    if (
        not np.isfinite(centroids).all()
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(
            "Ada angka embedding atau centroid tidak valid."
        )

    image_lookup = {
        str(image_id): index
        for index, image_id in enumerate(image_ids)
    }

    records = []

    for index in range(len(person_ids)):
        person_id = str(person_ids[index])
        full_name = str(person_names[index]).strip()

        selected_ids = [
            str(image_id)
            for image_id in reference_ids[index]
        ]

        if not full_name:
            raise ValueError(
                f"Nama kosong untuk {person_id}."
            )

        if len(set(selected_ids)) != 2:
            raise ValueError(
                f"Referensi harus dua gambar berbeda: {person_id}"
            )

        reference_vectors = []

        for image_id in selected_ids:
            if image_id not in image_lookup:
                raise ValueError(
                    f"Embedding tidak ditemukan: {image_id}"
                )

            position = image_lookup[image_id]

            if str(embedding_person_ids[position]) != person_id:
                raise ValueError(
                    f"Referensi bukan milik {person_id}: {image_id}"
                )

            reference_vectors.append(
                embeddings[position]
            )

        reference_vectors = np.stack(reference_vectors)

        if not np.allclose(
            np.linalg.norm(reference_vectors, axis=1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"Referensi belum ternormalisasi: {person_id}"
            )

        # Hitung ulang centroid untuk memeriksa kesesuaian.
        rebuilt_centroid = reference_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(rebuilt_centroid)

        if centroid_norm <= 1e-12:
            raise ValueError(
                f"Centroid tidak valid: {person_id}"
            )

        rebuilt_centroid = rebuilt_centroid / centroid_norm

        if (
            not np.isclose(
                np.linalg.norm(centroids[index]),
                1.0,
                atol=1e-5,
            )
            or not np.allclose(
                rebuilt_centroid,
                centroids[index],
                atol=1e-5,
                rtol=0,
            )
        ):
            raise ValueError(
                f"Centroid tidak sesuai referensi: {person_id}"
            )

        records.append({
            "person_id": person_id,
            "full_name": full_name,
            "centroid": centroids[index],
            "reference_ids": selected_ids,
            "reference_vectors": reference_vectors,
        })

    return records


def existing_person_matches(connection, record, existing):
    """Periksa bahwa data yang sudah tersimpan sama."""

    if (
        existing["full_name"] != record["full_name"]
        or not np.array_equal(
            np.asarray(
                existing["centroid"],
                dtype=np.float32,
            ),
            record["centroid"],
        )
    ):
        return False

    references = connection.execute(
        text("""
            SELECT
                reference_number,
                source_image_id,
                embedding
            FROM public.face_references
            WHERE person_id = :person_id
            ORDER BY reference_number
        """),
        {"person_id": record["person_id"]},
    ).mappings().all()

    if len(references) != 2:
        return False

    for index, reference in enumerate(references):
        if (
            reference["reference_number"] != index + 1
            or reference["source_image_id"]
            != record["reference_ids"][index]
            or not np.array_equal(
                np.asarray(
                    reference["embedding"],
                    dtype=np.float32,
                ),
                record["reference_vectors"][index],
            )
        ):
            return False

    return True


def main():
    print("Memulai impor galeri awal...", flush=True)
    print("File script:", Path(__file__).resolve(), flush=True)

    # Seluruh artefak diperiksa sebelum database diubah.
    records = load_initial_gallery()

    print(
        "Identitas siap diimpor:",
        len(records),
        flush=True,
    )
    print(
        "Referensi siap diimpor:",
        len(records) * 2,
        flush=True,
    )

    engine = get_engine()
    inserted = 0
    skipped = 0

    # Commit hanya jika seluruh proses dalam blok berhasil.
    with engine.begin() as connection:
        connection.execute(
            text("""
                LOCK TABLE
                    public.persons,
                    public.face_references
                IN SHARE ROW EXCLUSIVE MODE
            """)
        )

        for record in records:
            person_id = record["person_id"]

            existing = connection.execute(
                text("""
                    SELECT full_name, centroid
                    FROM public.persons
                    WHERE person_id = :person_id
                """),
                {"person_id": person_id},
            ).mappings().first()

            if existing is not None:
                if not existing_person_matches(
                    connection,
                    record,
                    existing,
                ):
                    raise ValueError(
                        f"Data lama berbeda untuk {person_id}. "
                        "Impor dibatalkan tanpa menimpa data."
                    )

                skipped += 1
                continue

            connection.execute(
                text("""
                    INSERT INTO public.persons (
                        person_id,
                        full_name,
                        centroid
                    )
                    VALUES (
                        :person_id,
                        :full_name,
                        CAST(:centroid AS REAL[])
                    )
                """),
                {
                    "person_id": person_id,
                    "full_name": record["full_name"],
                    "centroid": record["centroid"].tolist(),
                },
            )

            reference_rows = []

            for index in range(2):
                reference_rows.append({
                    "person_id": person_id,
                    "reference_number": index + 1,
                    "source_image_id": (
                        record["reference_ids"][index]
                    ),
                    "embedding": (
                        record["reference_vectors"][index].tolist()
                    ),
                })

            connection.execute(
                text("""
                    INSERT INTO public.face_references (
                        person_id,
                        reference_number,
                        source_image_id,
                        embedding
                    )
                    VALUES (
                        :person_id,
                        :reference_number,
                        :source_image_id,
                        CAST(:embedding AS REAL[])
                    )
                """),
                reference_rows,
            )

            inserted += 1

        total_persons = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM public.persons
            """)
        ).scalar_one()

        total_references = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM public.face_references
            """)
        ).scalar_one()

    print("\nImpor berhasil.", flush=True)
    print("Identitas baru ditambahkan :", inserted)
    print("Sudah ada dan sesuai       :", skipped)
    print("Total identitas di database:", total_persons)
    print("Total referensi di database:", total_references)


if __name__ == "__main__":
    main()