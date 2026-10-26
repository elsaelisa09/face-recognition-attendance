import hashlib
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen



MODEL_ZIP_URL = (
    "https://github.com/deepinsight/insightface/"
    "releases/download/v0.7/buffalo_l.zip"
)

MODEL_SPECS = {
    "detection_model": (
        "det_10g.onnx",
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    ),
    "recognition_model": (
        "w600k_r50.onnx",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    ),
}

_MODEL_LOCK = threading.Lock()


def _sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _download_archive(destination):
    request = Request(
        MODEL_ZIP_URL,
        headers={
            "User-Agent": "face-attendance-prototype",
        },
    )

    started = time.monotonic()
    total = 0
    next_progress = 25 * 1024 * 1024

    with urlopen(request, timeout=30) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                total += len(chunk)

                if total > 512 * 1024 * 1024:
                    raise RuntimeError(
                        "Paket model melebihi batas ukuran."
                    )

                if time.monotonic() - started > 600:
                    raise TimeoutError(
                        "Unduhan model melewati batas waktu."
                    )

                handle.write(chunk)

                if total >= next_progress:
                    print(
                        f"Unduhan model: "
                        f"{total / 1024**2:.0f} MiB",
                        flush=True,
                    )

                    next_progress += 25 * 1024 * 1024


def ensure_models(project_root, config):
    root = Path(project_root).resolve()

    with _MODEL_LOCK:
        targets = {}
        pending = []

        for key, (filename, expected_hash) in MODEL_SPECS.items():
            target = (root / config[key]).resolve()

            if (
                not target.is_relative_to(root)
                or target.name != filename
            ):
                raise ValueError(
                    f"Lokasi model tidak sesuai: {key}"
                )

            targets[key] = target

            if (
                not target.is_file()
                or _sha256(target) != expected_hash
            ):
                pending.append(key)

        if pending:
            work_root = root / "weights"
            work_root.mkdir(parents=True, exist_ok=True)

            print(
                "Mengunduh paket model buffalo_l...",
                flush=True,
            )

            with tempfile.TemporaryDirectory(
                dir=work_root
            ) as temporary:
                temporary = Path(temporary)
                archive_path = temporary / "buffalo_l.zip"

                _download_archive(archive_path)

                verified = {}

                with zipfile.ZipFile(archive_path) as archive:
                    for key in pending:
                        filename, expected_hash = MODEL_SPECS[key]

                        entries = [
                            entry
                            for entry in archive.infolist()
                            if not entry.is_dir()
                            and PurePosixPath(
                                entry.filename
                            ).name == filename
                        ]

                        if len(entries) != 1:
                            raise ValueError(
                                f"Isi paket tidak sesuai: {filename}"
                            )

                        entry = entries[0]

                        if not (
                            0 < entry.file_size
                            <= 256 * 1024 * 1024
                        ):
                            raise ValueError(
                                f"Ukuran model tidak valid: {filename}"
                            )

                        staged = temporary / filename

                        with (
                            archive.open(entry) as source,
                            staged.open("wb") as output,
                        ):
                            shutil.copyfileobj(
                                source,
                                output,
                                length=1024 * 1024,
                            )

                        if _sha256(staged) != expected_hash:
                            raise ValueError(
                                f"SHA256 model tidak sesuai: {filename}"
                            )

                        verified[key] = staged

                # Simpan setelah seluruh model yang diperlukan
                # berhasil diverifikasi.
                for key, staged in verified.items():
                    target = targets[key]
                    target.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    staged.replace(target)

                    print(
                        f"Model terverifikasi: {target.name}",
                        flush=True,
                    )

        return (
            targets["detection_model"],
            targets["recognition_model"],
        )