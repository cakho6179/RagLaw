import os
import json
import shutil
import urllib.parse
from pathlib import Path


def recover_files():
    history_dir = Path(r"C:\Users\NOS\AppData\Roaming\Code\User\History")
    target_dir = Path(r"d:\Project\RagLaw\backend")

    if not history_dir.exists():
        print("No VS Code history found!")
        return

    # Keep track of restored files to only restore the most recent
    restored = set()

    # We want to look at all folders in history
    folders = [f for f in history_dir.iterdir() if f.is_dir()]

    # Read entries.json from all folders to map file paths
    # We will sort by timestamp descending to get the latest non-empty version
    candidates = []

    for folder in folders:
        entries_file = folder / "entries.json"
        if not entries_file.exists():
            continue

        try:
            with open(entries_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            resource = data.get("resource", "")
            # Example resource: file:///d%3A/Project/RagLaw/backend/app.py
            if "RagLaw/backend" in urllib.parse.unquote(resource):
                # Local path
                raw_path = (
                    urllib.parse.unquote(resource)
                    .replace("file:///", "")
                    .replace("/", "\\")
                )
                if raw_path.startswith("d:"):
                    entries = data.get("entries", [])
                    # We want the most recent entry that has content
                    # We'll just collect them all
                    for entry in entries:
                        entry_file = folder / entry.get("id")
                        if entry_file.exists():
                            size = os.path.getsize(entry_file)
                            if size > 10:  # Ignore 0 byte or empty files
                                timestamp = entry.get("timestamp", 0)
                                candidates.append(
                                    {
                                        "target": raw_path,
                                        "source": entry_file,
                                        "timestamp": timestamp,
                                        "size": size,
                                    }
                                )
        except Exception as e:
            pass

    # Sort candidates by timestamp descending
    candidates.sort(key=lambda x: x["timestamp"], reverse=True)

    # Restore the most recent for each target
    count = 0
    for cand in candidates:
        target_path = cand["target"]
        if target_path not in restored:
            # Check if current file is empty or missing
            if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                # Restore
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(cand["source"], target_path)
                print(f"Restored: {target_path} ({cand['size']} bytes)")
                count += 1
            restored.add(target_path)

    print(f"\nSuccessfully recovered {count} files from VS Code history.")


if __name__ == "__main__":
    recover_files()
