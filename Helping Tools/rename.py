import os
import shutil
from pathlib import Path

def rename_and_move_images(source_folder, dest_folder="data", prefix="dummy", extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    source = Path(source_folder)
    destination = Path(dest_folder)

    if not source.is_dir():
        print(f"Error: Source directory '{source_folder}' does not exist.")
        return

    destination.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in source.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        print(f"No matching image files found in '{source_folder}'.")
        return

    image_files.sort(key=lambda x: x.name)

    print(f"Found {len(image_files)} images in '{source}'. Moving to '{destination}'...\n")

    for idx, file_path in enumerate(image_files, start=0):
        new_filename = f"{prefix}_{idx:04d}{file_path.suffix.lower()}"
        target_path = destination / new_filename

        shutil.move(str(file_path), str(target_path))
        print(f"Moved: {file_path.name} -> {target_path}")

    print(f"\nDone! Moved and renamed {len(image_files)} files into '{destination}'.")

if __name__ == "__main__":
    SOURCE_FOLDER = "dataa"
    DEST_FOLDER = "data"

    rename_and_move_images(
        source_folder=SOURCE_FOLDER,
        dest_folder=DEST_FOLDER,
        prefix="NOK" #OK/NOK MAKES THE FILE OK/NOK_x
    )
