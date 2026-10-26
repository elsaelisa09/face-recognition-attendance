"""Logic checks with synthetic data; these do not validate real recognition accuracy."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.face_pipeline import FaceRecognitionPipeline


class FakeDetector:
    def __init__(self):
        self.count = 1

    def prepare(self, **kwargs):
        self.options = kwargs

    def detect(self, image, max_num=0):
        boxes = np.array([[10, 10, 90, 95, 0.99], [110, 10, 190, 95, 0.9]], dtype=np.float32)
        points = np.array([[[30, 35], [65, 35], [48, 55], [35, 75], [60, 75]],
                           [[130, 35], [165, 35], [148, 55], [135, 75], [160, 75]]], dtype=np.float32)
        return boxes[:self.count], points[:self.count]


class FakeEncoder:
    input_size = (112, 112)

    def __init__(self):
        self.vector = np.array([1, 0, 0], dtype=np.float32)
        self.calls = 0

    def prepare(self, **kwargs):
        pass

    def get_feat(self, image):
        self.calls += 1
        self.last_image = image.copy()
        return self.vector[None, :].copy()


class PipelineLogicChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "artifacts").mkdir()
        (self.root / "weights").mkdir()
        for name in ("det_10g.onnx", "w600k_r50.onnx"):
            (self.root / "weights" / name).touch()
        gallery = self.root / "artifacts" / "gallery_70.npz"
        np.savez_compressed(
            gallery,
            centroids=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            person_ids=np.array(["person_001", "person_002"], dtype=str),
            person_names=np.array(["Person One", "Person Two"], dtype=str),
        )
        config = {
            "method": "centroid", "recognition_threshold": 0.47,
            "recognition_threshold_status": "provisional_for_70_identities",
            "gallery_file": "artifacts/gallery_70.npz", "n_identities": 2,
            "aligned_image_size": 112, "detection_model": "weights/det_10g.onnx",
            "recognition_model": "weights/w600k_r50.onnx",
            "detector_input_size": [640, 640], "detector_threshold": 0.5,
            "gallery_sha256": hashlib.sha256(gallery.read_bytes()).hexdigest(),
        }
        (self.root / "artifacts" / "inference_config.json").write_text(json.dumps(config))
        self.detector = FakeDetector()
        self.encoder = FakeEncoder()

        def loader(path, **kwargs):
            self.assertEqual(kwargs["providers"], ["CPUExecutionProvider"])
            return self.detector if Path(path).name == "det_10g.onnx" else self.encoder

        self.loader = loader
        self.selected_landmarks = []

        def aligner(image, landmark, image_size, mode):
            self.selected_landmarks.append(landmark.copy())
            return np.resize(image, (image_size, image_size, 3))

        self.aligner = aligner
        self.pipeline = FaceRecognitionPipeline(
            self.root, model_loader=loader, aligner=aligner
        )
        self.image = Image.new("RGB", (220, 160), (255, 0, 0))

    def test_unicode_gallery_and_rgb_bgr_consistency(self):
        result = self.pipeline.predict(self.image)
        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["person_id"], "person_001")
        self.assertEqual(result["aligned_face"].size, (112, 112))
        np.testing.assert_array_equal(self.encoder.last_image[0, 0], [0, 0, 255])

    def test_embedding_normalization(self):
        self.encoder.vector = np.array([7, 0, 0], dtype=np.float32)
        result = self.pipeline.predict(self.image)
        self.assertAlmostEqual(result["similarity"], 1.0)

    def test_unknown_does_not_return_an_accepted_identity(self):
        self.encoder.vector = np.array([0, 0, 1], dtype=np.float32)
        result = self.pipeline.predict(self.image)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["person_name"])
        self.assertIsNone(result["person_id"])

    def test_no_face_does_not_run_encoder(self):
        self.detector.count = 0
        result = self.pipeline.predict(self.image)
        self.assertEqual(result["status"], "no_face")
        self.assertEqual(self.encoder.calls, 0)

    def test_multiple_faces_require_explicit_selection(self):
        self.detector.count = 2
        result = self.pipeline.predict(self.image)
        self.assertEqual(result["status"], "select_face")
        self.assertEqual(self.encoder.calls, 0)
        result = self.pipeline.predict(self.image, face_index=1)
        self.assertEqual(result["face_index"], 1)
        self.assertEqual(float(self.selected_landmarks[-1][0, 0]), 130)
        with self.assertRaises(ValueError):
            self.pipeline.predict(self.image, face_index=9)

    def test_zero_embedding_is_rejected(self):
        self.encoder.vector = np.zeros(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.pipeline.predict(self.image)

    def test_gallery_hash_mismatch_is_rejected(self):
        gallery_path = self.root / "artifacts" / "gallery_70.npz"
        gallery_path.write_bytes(gallery_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            FaceRecognitionPipeline(
                self.root, model_loader=self.loader, aligner=self.aligner
            )

    def test_exif_orientation_is_applied(self):
        image = Image.new("RGB", (20, 10))
        image.getexif()[274] = 6
        self.assertEqual(self.pipeline.read_image(image).size, (10, 20))


if __name__ == "__main__":
    unittest.main()
