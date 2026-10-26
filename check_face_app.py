"""Check local model initialization and optionally run one image."""

import argparse
import json
import time
from pathlib import Path

from src.face_pipeline import FaceRecognitionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--face-index", type=int)
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    pipeline = FaceRecognitionPipeline(Path(__file__).resolve().parent)
    print("Pipeline siap.")
    print("Jumlah identitas:", len(pipeline.person_ids))
    print("Ukuran input:", pipeline.encoder.input_size)
    print("Threshold:", pipeline.threshold)
    print("Status threshold:", pipeline.threshold_status)
    print(f"Waktu pemuatan: {time.perf_counter() - started:.2f} detik")
    if args.image is not None:
        result = (
            pipeline.recognize_aligned(args.image) if args.aligned
            else pipeline.predict(args.image, face_index=args.face_index)
        )
        printable = {key: value for key, value in result.items()
                     if key not in ("image", "aligned_face")}
        print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
