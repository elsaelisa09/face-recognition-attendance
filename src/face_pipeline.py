
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from src.model_files import ensure_models


def _load_database_gallery():
    from sqlalchemy import text
    from src.database import get_engine

    with get_engine().connect() as connection:
        rows = connection.execute(text("""
            SELECT person_id, full_name, centroid
            FROM public.persons
            WHERE is_active = TRUE
            ORDER BY person_id
        """)).mappings().all()

    if not rows:
        raise ValueError(
            "Belum ada identitas aktif di database."
        )

    centroids = np.asarray(
        [row["centroid"] for row in rows],
        dtype=np.float32,
    )

    person_ids = np.asarray(
        [row["person_id"] for row in rows],
        dtype=str,
    )

    person_names = np.asarray(
        [row["full_name"] for row in rows],
        dtype=str,
    )

    if (
        centroids.shape != (len(rows), 512)
        or not np.isfinite(centroids).all()
        or len(np.unique(person_ids)) != len(rows)
    ):
        raise ValueError(
            "Struktur galeri database tidak valid."
        )

    norms = np.linalg.norm(centroids, axis=1)

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError(
            "Centroid database belum ternormalisasi."
        )

    return (
        centroids / norms[:, None],
        person_ids,
        person_names,
    )

class FaceRecognitionPipeline:
    def __init__(self, project_root=None, *, model_loader=None, aligner=None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.config_path = self.root / "artifacts" / "inference_config.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.config.get("method") != "centroid":
            raise ValueError("Konfigurasi harus menggunakan metode centroid.")
        self.threshold = float(self.config["recognition_threshold"])
        if not np.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Threshold pengenalan tidak valid.")
        self.threshold_status = self.config.get(
            "recognition_threshold_status", "unspecified"
        )
        self.centroids, self.person_ids, self.person_names = (
            _load_database_gallery()
        )
        self.image_size = int(self.config["aligned_image_size"])
        detector_path, encoder_path = ensure_models(
            self.root,
            self.config,
        )
        if model_loader is None:
            from insightface import model_zoo

            model_loader = model_zoo.get_model
        if aligner is None:
            from insightface.utils.face_align import norm_crop

            aligner = norm_crop
        self._aligner = aligner
        self._lock = threading.RLock()
        providers = ["CPUExecutionProvider"]
        self.detector = model_loader(str(detector_path), providers=providers)
        self.encoder = model_loader(str(encoder_path), providers=providers)
        if self.detector is None or self.encoder is None:
            raise ValueError("File ONNX tidak dikenali sebagai model InsightFace.")
        self.detector.prepare(
            ctx_id=-1,
            input_size=tuple(self.config["detector_input_size"]),
            det_thresh=float(self.config["detector_threshold"]),
        )
        self.encoder.prepare(ctx_id=-1)
        if tuple(self.encoder.input_size) != (self.image_size, self.image_size):
            raise ValueError("Ukuran input encoder berbeda dari konfigurasi.")

    @staticmethod
    def read_image(source):
        """Read the primary frame, apply EXIF orientation, and return an RGB copy."""
        if isinstance(source, Image.Image):
            return ImageOps.exif_transpose(source).convert("RGB")
        if hasattr(source, "seek"):
            source.seek(0)
        with Image.open(source) as image:
            image.seek(0)
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")

    @staticmethod
    def _bgr(image):
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()

    def extract_face_embedding(self, image, face):
        """Menghasilkan embedding tanpa mencocokkan ke galeri."""

        started = time.perf_counter()

        landmarks = np.asarray(
            face["landmarks"],
            dtype=np.float32,
        )

        if (
            landmarks.shape != (5, 2)
            or not np.isfinite(landmarks).all()
        ):
            raise ValueError("Landmark wajah tidak valid.")

        image_bgr = self._bgr(image)

        aligned_bgr = self._aligner(
            image_bgr,
            landmark=landmarks,
            image_size=self.image_size,
            mode="arcface",
        )

        expected_shape = (
            self.image_size,
            self.image_size,
            3,
        )

        if aligned_bgr.shape != expected_shape:
            raise ValueError(
                "Ukuran hasil alignment tidak sesuai."
            )

        with self._lock:
            embedding = np.asarray(
                self.encoder.get_feat(aligned_bgr),
                dtype=np.float32,
            ).reshape(-1)

        if (
            embedding.shape != (512,)
            or not np.isfinite(embedding).all()
        ):
            raise ValueError(
                "Model menghasilkan embedding tidak valid."
            )

        norm = np.linalg.norm(embedding)

        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(
                "Embedding tidak dapat dinormalisasi."
            )

        embedding = embedding / norm

        aligned_face = Image.fromarray(
            aligned_bgr[:, :, ::-1].copy()
        )

        return {
            "embedding": embedding,
            "aligned_face": aligned_face,
            "face_index": int(face["face_index"]),
            "processing_ms": (
                time.perf_counter() - started
            ) * 1000,
        }

    def detect_faces(self, image):
        """Return all usable detections; indices refer only to this input image."""
        image_bgr = self._bgr(image)
        with self._lock:
            boxes, points = self.detector.detect(image_bgr, max_num=0)
        if boxes is None or len(boxes) == 0:
            return []
        if points is None or len(points) != len(boxes):
            raise ValueError("Detector tidak menghasilkan landmark yang sesuai.")
        faces = []
        for box, landmarks in zip(boxes, points):
            box = np.asarray(box, dtype=np.float32)
            landmarks = np.asarray(landmarks, dtype=np.float32)
            if (
                box.size < 5
                or landmarks.shape != (5, 2)
                or not np.isfinite(box).all()
                or not np.isfinite(landmarks).all()
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                continue
            faces.append({
                "face_index": len(faces),
                "bbox": box[:4].tolist(),
                "landmarks": landmarks.tolist(),
                "det_score": float(box[4]),
            })
        return faces

    def _match(self, aligned_bgr):
        centroids, person_ids, person_names = (
            _load_database_gallery()
        )

        with self._lock:
            vector = np.asarray(
                self.encoder.get_feat(aligned_bgr),
                dtype=np.float32,
            ).reshape(-1)

        length = np.linalg.norm(vector)

        if (
            vector.size != centroids.shape[1]
            or not np.isfinite(vector).all()
            or not np.isfinite(length)
            or length <= 1e-12
        ):
            raise ValueError(
                "Encoder menghasilkan embedding tidak valid."
            )

        vector = vector / length

        scores = np.clip(
            centroids @ vector,
            -1.0,
            1.0,
        )

        order = np.argsort(-scores, kind="stable")
        best = int(order[0])

        similarity = float(scores[best])
        accepted = similarity >= self.threshold

        return {
            "status": "recognized" if accepted else "unknown",
            "accepted": bool(accepted),
            "person_id": (
                str(person_ids[best]) if accepted else None
            ),
            "person_name": (
                str(person_names[best]) if accepted else None
            ),
            "candidate_person_id": str(person_ids[best]),
            "candidate_person_name": str(person_names[best]),
            "similarity": similarity,
            "threshold": self.threshold,
            "threshold_status": self.threshold_status,
            "top_matches": [
                {
                    "person_id": str(person_ids[index]),
                    "person_name": str(person_names[index]),
                    "similarity": float(scores[index]),
                }
                for index in order[:3]
            ],
        }

    def recognize_face(self, image, face):
        """Recognize a selected detection using its landmarks on this same image."""
        started = time.perf_counter()
        landmarks = np.asarray(face["landmarks"], dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("Landmark wajah tidak valid.")
        image_bgr = self._bgr(image)
        aligned_bgr = self._aligner(
            image_bgr, landmark=landmarks,
            image_size=self.image_size, mode="arcface"
        )
        if aligned_bgr.shape != (self.image_size, self.image_size, 3):
            raise ValueError("Ukuran hasil alignment tidak sesuai.")
        result = self._match(aligned_bgr)
        result["face_index"] = int(face["face_index"])
        result["aligned_face"] = Image.fromarray(aligned_bgr[:, :, ::-1].copy())
        result["recognition_ms"] = (time.perf_counter() - started) * 1000
        return result

    def recognize_aligned(self, source):
        """Recognize an existing aligned crop without detecting or aligning again."""
        started = time.perf_counter()
        image = self.read_image(source)
        if image.size != (self.image_size, self.image_size):
            raise ValueError("Crop harus memiliki ukuran input encoder.")
        result = self._match(self._bgr(image))
        result["aligned_face"] = image
        result["recognition_ms"] = (time.perf_counter() - started) * 1000
        return result

    def predict(self, source, face_index=None):
        """Convenience API: require selection when multiple faces are detected."""
        started = time.perf_counter()
        image = self.read_image(source)
        detection_started = time.perf_counter()
        faces = self.detect_faces(image)
        detection_ms = (time.perf_counter() - detection_started) * 1000
        base = {"image": image, "faces": faces, "n_faces": len(faces),
                "detection_ms": detection_ms}
        if not faces:
            result = {**base, "status": "no_face"}
        elif len(faces) > 1 and face_index is None:
            result = {**base, "status": "select_face"}
        else:
            chosen = 0 if face_index is None else face_index
            if isinstance(chosen, bool) or not isinstance(chosen, (int, np.integer)):
                raise ValueError("Nomor wajah harus berupa bilangan bulat.")
            selected = next((face for face in faces
                             if face["face_index"] == chosen), None)
            if selected is None:
                raise ValueError("Nomor wajah tidak tersedia pada foto ini.")
            result = {**base, **self.recognize_face(image, selected)}
        result["total_ms"] = (time.perf_counter() - started) * 1000
        return result




def draw_faces(image, faces, selected_index=None, max_size=(1100, 850)):
    """Draw numbered boxes on a preview without modifying recognition pixels."""
    preview = image.convert("RGB").copy()
    original_width, original_height = preview.size
    preview.thumbnail(max_size)
    sx, sy = preview.width / original_width, preview.height / original_height
    draw = ImageDraw.Draw(preview)
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        color = "#15803d" if face["face_index"] == selected_index else "#dc2626"
        draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=color, width=3)
        left = min(max(0, int(x1 * sx)), max(0, preview.width - 80))
        top = min(max(0, int(y1 * sy) - 23), max(0, preview.height - 23))
        draw.rectangle((left, top, left + 80, top + 23), fill=color)
        draw.text((left + 5, top + 5), f"face {face['face_index']}", fill="white")
    return preview
