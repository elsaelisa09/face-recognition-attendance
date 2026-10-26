import hashlib
import time
from datetime import timedelta, timezone
from uuid import uuid4

import numpy as np
from sqlalchemy import text

from src.database import get_engine


class RegistrationError(ValueError):
    """Kesalahan pada data pendaftaran."""


class AttendanceError(ValueError):
    """Kesalahan saat memproses absensi."""

class DuplicateFaceError(RegistrationError):
    def __init__(self, full_name, similarity, threshold):
        self.full_name = str(full_name)
        self.similarity = float(similarity)
        self.threshold = float(threshold)

        super().__init__(
            "Wajah terindikasi sudah terdaftar. "
            "Pendaftaran baru tidak disimpan."
        )

def ensure_face_not_registered(
    reference_embeddings,
    gallery,
    threshold,
):
    if not gallery:
        return

    threshold = float(threshold)

    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise RegistrationError(
            "Ambang pemeriksaan duplikat tidak valid."
        )

    queries = np.asarray(
        reference_embeddings,
        dtype=np.float32,
    )

    if queries.shape != (2, 512):
        raise RegistrationError(
            "Pemeriksaan membutuhkan dua embedding wajah."
        )

    # Normalisasi sekaligus memeriksa setiap embedding.
    queries = np.stack([
        normalize_embedding(vector)
        for vector in queries
    ])

    centroids = np.stack([
        normalize_embedding(person["centroid"])
        for person in gallery
    ])

    # Bandingkan masing-masing foto dengan seluruh identitas.
    scores = np.clip(
        queries @ centroids.T,
        -1.0,
        1.0,
    )

    photo_index, person_index = np.unravel_index(
        np.argmax(scores),
        scores.shape,
    )

    similarity = float(scores[photo_index, person_index])

    if similarity >= threshold:
        raise DuplicateFaceError(
            full_name=gallery[int(person_index)]["full_name"],
            similarity=similarity,
            threshold=threshold,
        )

def register_person(full_name, photo_sources, pipeline):
    prepared = prepare_registration(
        full_name,
        photo_sources,
        pipeline,
    )

    person_id = f"person_{uuid4().hex}"
    threshold = float(pipeline.threshold)
    engine = get_engine()

    with engine.begin() as connection:
        # Pemeriksaan dan penyimpanan pendaftaran dilakukan bergiliran.
        connection.execute(
            text("""
                LOCK TABLE public.persons
                IN SHARE ROW EXCLUSIVE MODE
            """)
        )

        gallery = connection.execute(
            text("""
                SELECT person_id, full_name, centroid
                FROM public.persons
                ORDER BY person_id
            """)
        ).mappings().all()

        # Jika terindikasi duplikat, fungsi ini melempar error
        # sebelum ada identitas baru yang disimpan.
        ensure_face_not_registered(
            reference_embeddings=prepared[
                "reference_embeddings"
            ],
            gallery=gallery,
            threshold=threshold,
        )

        connection.execute(
            text("""
                INSERT INTO public.persons (
                    person_id,
                    full_name,
                    centroid,
                    is_active
                )
                VALUES (
                    :person_id,
                    :full_name,
                    CAST(:centroid AS REAL[]),
                    TRUE
                )
            """),
            {
                "person_id": person_id,
                "full_name": prepared["full_name"],
                "centroid": prepared["centroid"].tolist(),
            },
        )

        reference_rows = [
            {
                "person_id": person_id,
                "reference_number": index + 1,
                "source_image_id": (
                    f"sha256:{prepared['image_hashes'][index]}"
                ),
                "embedding": (
                    prepared["reference_embeddings"][index].tolist()
                ),
            }
            for index in range(2)
        ]

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

    return {
        "status": "registered",
        "person_id": person_id,
        "full_name": prepared["full_name"],
        "reference_count": 2,
        "aligned_faces": prepared["aligned_faces"],
    }


def normalize_embedding(value):
    embedding = np.asarray(
        value,
        dtype=np.float32,
    ).reshape(-1)

    if (
        embedding.shape != (512,)
        or not np.isfinite(embedding).all()
    ):
        raise RegistrationError(
            "Embedding wajah tidak valid."
        )

    norm = np.linalg.norm(embedding)

    if not np.isfinite(norm) or norm <= 1e-12:
        raise RegistrationError(
            "Embedding tidak dapat dinormalisasi."
        )

    return embedding / norm


def prepare_registration(full_name, photo_sources, pipeline):
    if not isinstance(full_name, str):
        raise RegistrationError(
            "Nama lengkap harus berupa teks."
        )

    full_name = " ".join(full_name.split())

    if not full_name:
        raise RegistrationError(
            "Nama lengkap wajib diisi."
        )

    if len(full_name) > 200:
        raise RegistrationError(
            "Nama lengkap maksimal 200 karakter."
        )

    if (
        not isinstance(photo_sources, (list, tuple))
        or len(photo_sources) != 2
    ):
        raise RegistrationError(
            "Unggah tepat dua foto dari orang yang sama."
        )

    embeddings = []
    image_hashes = []
    aligned_faces = []

    for number, source in enumerate(photo_sources, start=1):
        try:
            image = pipeline.read_image(source)

        except Exception as exc:
            raise RegistrationError(
                f"Foto {number} tidak dapat dibaca. "
                "Coba gambar lain."
            ) from exc

        digest = hashlib.sha256()

        digest.update(
            f"{image.width}x{image.height}:RGB:".encode("utf-8")
        )
        digest.update(image.tobytes())

        image_hash = digest.hexdigest()

        if image_hash in image_hashes:
            raise RegistrationError(
                "Kedua foto memiliki isi gambar yang sama. "
                "Gunakan dua foto dari pengambilan berbeda."
            )

        try:
            faces = pipeline.detect_faces(image)

            if len(faces) != 1:
                raise RegistrationError(
                    f"Foto {number} harus berisi tepat satu wajah; "
                    f"terdeteksi {len(faces)} wajah."
                )

            extracted = pipeline.extract_face_embedding(
                image,
                faces[0],
            )

            embedding = normalize_embedding(
                extracted["embedding"]
            )

        except RegistrationError:
            raise

        except Exception as exc:
            raise RegistrationError(
                f"Wajah pada foto {number} gagal diproses."
            ) from exc

        embeddings.append(embedding)
        image_hashes.append(image_hash)
        aligned_faces.append(extracted["aligned_face"])

    reference_embeddings = np.stack(embeddings)

    centroid = normalize_embedding(
        reference_embeddings.mean(axis=0)
    )

    return {
        "full_name": full_name,
        "centroid": centroid,
        "reference_embeddings": reference_embeddings,
        "image_hashes": image_hashes,
        "aligned_faces": aligned_faces,
    }



def mark_attendance(photo_source, pipeline, *, selected_face=None):
    started = time.perf_counter()

    try:
        image = pipeline.read_image(photo_source)

        if selected_face is None:
            faces = pipeline.detect_faces(image)

            if not faces:
                raise AttendanceError(
                    "Tidak ada wajah terdeteksi pada foto."
                )

            if len(faces) > 1:
                raise AttendanceError(
                    "Ada beberapa wajah. "
                    "Pilih nomor wajah untuk absensi."
                )

            selected_face = faces[0]

        if not isinstance(selected_face, dict):
            raise AttendanceError(
                "Pilihan wajah tidak valid."
            )

        extracted = pipeline.extract_face_embedding(
            image,
            selected_face,
        )

        query_embedding = normalize_embedding(
            extracted["embedding"]
        )

    except AttendanceError:
        raise

    except Exception as exc:
        raise AttendanceError(
            "Wajah yang dipilih gagal diproses. "
            "Coba foto lain."
        ) from exc

    engine = get_engine()

    # Membaca galeri terbaru, termasuk identitas yang baru didaftarkan.
    with engine.connect() as connection:
        gallery = connection.execute(
            text("""
                SELECT person_id, full_name, centroid
                FROM public.persons
                WHERE is_active = TRUE
                ORDER BY person_id
            """)
        ).mappings().all()

    if not gallery:
        raise AttendanceError(
            "Belum ada identitas aktif di database."
        )

    centroids = np.asarray(
        [person["centroid"] for person in gallery],
        dtype=np.float32,
    )

    if (
        centroids.shape != (len(gallery), 512)
        or not np.isfinite(centroids).all()
    ):
        raise AttendanceError(
            "Data centroid database tidak valid."
        )

    norms = np.linalg.norm(centroids, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise AttendanceError(
            "Centroid database belum ternormalisasi."
        )

    matching_started = time.perf_counter()

    centroids = centroids / norms[:, None]

    scores = np.clip(
        centroids @ query_embedding,
        -1.0,
        1.0,
    )

    best_index = int(np.argmax(scores))
    similarity = float(scores[best_index])
    threshold = float(pipeline.threshold)
    accepted = similarity >= threshold

    recognition_ms = (
        extracted["processing_ms"]
        + (time.perf_counter() - matching_started) * 1000
    )

    result = {
        "status": "recognized" if accepted else "unknown",
        "accepted": accepted,
        "person_id": None,
        "person_name": None,
        "similarity": similarity,
        "threshold": threshold,
        "aligned_face": extracted["aligned_face"],
        "recognition_ms": recognition_ms,
        "gallery_size": len(gallery),
        "attendance_status": "not_recorded",
    }

    # Wajah yang tidak lolos threshold tidak dicatat.
    if not accepted:
        result["total_ms"] = (
            time.perf_counter() - started
        ) * 1000

        return result

    person_id = str(
        gallery[best_index]["person_id"]
    )

    with engine.begin() as connection:
        full_name = connection.execute(
            text("""
                SELECT full_name
                FROM public.persons
                WHERE person_id = :person_id
                  AND is_active = TRUE
                FOR SHARE
            """),
            {"person_id": person_id},
        ).scalar_one_or_none()

        if full_name is None:
            raise AttendanceError(
                "Identitas sudah tidak aktif. Coba lagi."
            )

        attendance = connection.execute(
            text("""
                INSERT INTO public.attendance (
                    person_id,
                    similarity,
                    threshold_used
                )
                VALUES (
                    :person_id,
                    :similarity,
                    :threshold_used
                )
                ON CONFLICT ON CONSTRAINT
                    uq_person_attendance_date
                DO NOTHING
                RETURNING
                    attendance_id,
                    checked_in_at,
                    attendance_date
            """),
            {
                "person_id": person_id,
                "similarity": similarity,
                "threshold_used": threshold,
            },
        ).mappings().first()

        if attendance is not None:
            attendance_status = "recorded"

        else:
            attendance_status = "already_recorded"

            attendance = connection.execute(
                text("""
                    SELECT
                        attendance_id,
                        checked_in_at,
                        attendance_date
                    FROM public.attendance
                    WHERE person_id = :person_id
                      AND attendance_date = (
                          NOW() AT TIME ZONE 'Asia/Jakarta'
                      )::date
                """),
                {"person_id": person_id},
            ).mappings().one()

    wib = timezone(timedelta(hours=7))

    result.update({
        "person_id": person_id,
        "person_name": full_name,
        "attendance_status": attendance_status,
        "attendance_id": attendance["attendance_id"],
        "attendance_date": str(
            attendance["attendance_date"]
        ),
        "checked_in_at_wib": (
            attendance["checked_in_at"]
            .astimezone(wib)
            .strftime("%d-%m-%Y %H:%M:%S WIB")
        ),
        "total_ms": (
            time.perf_counter() - started
        ) * 1000,
    })

    return result

