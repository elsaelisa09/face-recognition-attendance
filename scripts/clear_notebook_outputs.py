"""Clear only the public notebook copies before sharing them."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "notebooks_public"
    paths = sorted(folder.glob("*.ipynb"))
    if not paths:
        raise SystemExit("Tidak ada notebook di notebooks_public/.")

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["metadata"] = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
        for cell in notebook.get("cells", []):
            cell["metadata"] = {}
            cell.pop("attachments", None)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None

        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        print("Output dibersihkan:", path.relative_to(root))


if __name__ == "__main__":
    main()
