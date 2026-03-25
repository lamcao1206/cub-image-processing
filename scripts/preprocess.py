import logging
import re
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "CUB_200_2011"
IMAGES_DIR = DATA_DIR / "images"
SEGMENTATIONS_DIR = DATA_DIR / "segmentations"
CLASSES_TXT = DATA_DIR / "classes.txt"
IMAGE_TXT = DATA_DIR / "images.txt"

LOGGER = logging.getLogger("data_preprocessing")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def clean_label(raw_label: str) -> str:
    label = re.sub(r"^\d+[._-]*", "", raw_label)
    label = label.replace(".", "_")
    label = re.sub(r"[^A-Za-z0-9]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label.lower()


def scan_disk_class_mapping(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    if not root.exists():
        LOGGER.error("Directory does not exist: %s", root)
        return mapping

    used: Dict[str, int] = {}

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue

        old_name = d.name
        candidate = clean_label(old_name)

        if not candidate:
            LOGGER.warning("Skipping invalid folder name: %s", old_name)
            continue

        if candidate in used:
            used[candidate] += 1
            candidate = f"{candidate}_{used[candidate]}"
            LOGGER.warning("Duplicate detected, renamed to: %s", candidate)
        else:
            used[candidate] = 1

        mapping[old_name] = candidate

    LOGGER.info("Scanned %d classes in %s", len(mapping), root)
    return mapping


def validate_mapping(mapping: Dict[str, str], name: str):
    LOGGER.info("Validating %s...", name)

    total = len(mapping)
    unique = len(set(mapping.values()))

    LOGGER.info("%s total classes: %d", name, total)
    LOGGER.info("%s unique labels: %d", name, unique)

    if total != 200:
        LOGGER.error("❌ %s does NOT have 200 classes!", name)
    else:
        LOGGER.info("✅ %s has 200 classes", name)

    if total != unique:
        LOGGER.error("❌ %s has duplicate labels!", name)
    else:
        LOGGER.info("✅ %s has no duplicates", name)


def apply_folder_renaming(root: Path, mapping: Dict[str, str], dry_run=True):
    LOGGER.info("Starting rename in: %s (dry_run=%s)", root, dry_run)

    temp_paths = {}

    for old_name, new_name in mapping.items():
        LOGGER.info("PLAN: %s -> %s", old_name, new_name)

    if dry_run:
        LOGGER.info("Dry run enabled. No changes applied.")
        return

    for old_name, new_name in mapping.items():
        src = root / old_name

        if not src.exists():
            LOGGER.warning("Missing folder: %s", src)
            continue

        tmp = root / f"__tmp__{new_name}"

        LOGGER.info("Renaming (step 1): %s -> %s", src.name, tmp.name)
        src.rename(tmp)

        temp_paths[tmp] = root / new_name

    for tmp, final in temp_paths.items():
        if final.exists():
            LOGGER.error("Target already exists: %s", final)
            continue

        LOGGER.info("Renaming (step 2): %s -> %s", tmp.name, final.name)
        tmp.rename(final)

    LOGGER.info("✅ Completed renaming in: %s", root)


def transform_classes_txt_inplace(file_path: Path):
    LOGGER.info("Transforming classes.txt IN-PLACE: %s", file_path)

    if not file_path.exists():
        LOGGER.error("File not found: %s", file_path)
        return

    # 🔒 Step 0: backup
    backup_path = file_path.with_suffix(".backup.txt")
    file_path.replace(backup_path)
    LOGGER.info("Backup created: %s", backup_path)

    cleaned_labels = []

    # Read from backup
    with backup_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2:
                _, raw_label = parts
            else:
                raw_label = parts[0]

            cleaned = clean_label(raw_label)
            cleaned_labels.append(cleaned)

            LOGGER.info("CLASS: %s -> %s", raw_label, cleaned)

    # Validation
    if len(cleaned_labels) != 200:
        LOGGER.error(
            "❌ classes.txt does NOT have 200 classes! Found: %d", len(cleaned_labels)
        )
    else:
        LOGGER.info("✅ classes.txt has 200 classes")

    if len(set(cleaned_labels)) != len(cleaned_labels):
        LOGGER.error("❌ Duplicate labels found!")
    else:
        LOGGER.info("✅ No duplicates in cleaned labels")

    # ✍️ Write BACK to original file
    with file_path.open("w") as f:
        for i, label in enumerate(cleaned_labels):
            f.write(f"{i} {label}\n")

    LOGGER.info("✅ classes.txt overwritten successfully")


def rename_images_txt_only(images_txt: Path):
    LOGGER.info("Rewriting images.txt ONLY (no disk changes): %s", images_txt)

    if not images_txt.exists():
        LOGGER.error("images.txt not found!")
        return

    lines_out = []

    with images_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            image_id, path = line.split(" ", 1)

            # split folder + filename
            folder, filename = path.split("/", 1)

            # 🔥 clean ONLY the folder name
            new_folder = clean_label(folder)

            new_path = f"{new_folder}/{filename}"

            lines_out.append(f"{image_id} {new_path}")

    # overwrite file
    with images_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    LOGGER.info("✅ Done rewriting images.txt")


def main():
    configure_logging()

    LOGGER.info("Data directory: %s", DATA_DIR)
    LOGGER.info("Images directory: %s", IMAGES_DIR)
    LOGGER.info("Segmentations directory: %s", SEGMENTATIONS_DIR)

    disk_image_map = scan_disk_class_mapping(IMAGES_DIR)
    disk_seg_map = scan_disk_class_mapping(SEGMENTATIONS_DIR)

    validate_mapping(disk_image_map, "Images")
    validate_mapping(disk_seg_map, "Segmentations")

    # apply_folder_renaming(IMAGES_DIR, disk_image_map, dry_run=True)
    # apply_folder_renaming(SEGMENTATIONS_DIR, disk_seg_map, dry_run=True)

    apply_folder_renaming(IMAGES_DIR, disk_image_map, dry_run=False)
    apply_folder_renaming(SEGMENTATIONS_DIR, disk_seg_map, dry_run=False)
    transform_classes_txt_inplace(CLASSES_TXT)
    rename_images_txt_only(IMAGE_TXT)


if __name__ == "__main__":
    main()
