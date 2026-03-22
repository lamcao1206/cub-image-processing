import argparse
import csv
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


LOGGER = logging.getLogger("data_preprocessing")


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    rel_path: str


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def clean_label(raw_label: str) -> str:
    label = re.sub(r"^\d+[._-]*", "", raw_label)
    label = label.replace(".", "_")
    label = re.sub(r"[^A-Za-z0-9]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label.lower()


def read_classes(classes_file: Path) -> Dict[int, str]:
    classes: Dict[int, str] = {}
    with classes_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id_str, class_name = line.split(" ", 1)
            classes[int(class_id_str)] = class_name
    return classes


def make_class_mapping(classes: Dict[int, str]) -> Dict[str, str]:
    used: Dict[str, int] = {}
    old_to_new: Dict[str, str] = {}

    for class_id in sorted(classes.keys()):
        old_name = classes[class_id]
        candidate = clean_label(old_name)
        if not candidate:
            candidate = f"class_{class_id}"

        if candidate in used:
            used[candidate] += 1
            candidate = f"{candidate}_{used[candidate]}"
        else:
            used[candidate] = 1

        old_to_new[old_name] = candidate

    return old_to_new


def read_images(images_file: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    with images_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, rel_path = line.split(" ", 1)
            records.append(ImageRecord(image_id=int(image_id_str), rel_path=rel_path))
    return records


def stem_with_new_label(stem: str, old_label: str, new_label: str) -> str:
    old_simple = clean_label(re.sub(r"^\d+[._-]*", "", old_label))

    match = re.match(r"^(.*?)(_\d+(?:_\d+)*)$", stem)
    if match:
        prefix, suffix = match.groups()
        if clean_label(prefix) == old_simple:
            return f"{new_label}{suffix}"

    if clean_label(stem).startswith(old_simple):
        return new_label

    return stem


def build_new_image_paths(
    records: List[ImageRecord],
    class_map: Dict[str, str],
) -> Dict[int, str]:
    new_paths: Dict[int, str] = {}

    for rec in records:
        rel = Path(rec.rel_path)
        old_class = rel.parts[0]
        old_name = rel.stem
        suffix = rel.suffix
        new_class = class_map[old_class]
        new_name = stem_with_new_label(old_name, old_class, new_class)
        new_rel = Path(new_class) / f"{new_name}{suffix}"
        new_paths[rec.image_id] = new_rel.as_posix()

    return new_paths


def write_classes(classes_file: Path, classes: Dict[int, str], class_map: Dict[str, str], apply: bool) -> None:
    output = []
    for class_id in sorted(classes.keys()):
        output.append(f"{class_id} {class_map[classes[class_id]]}\n")

    if apply:
        classes_file.write_text("".join(output), encoding="utf-8")


def write_images(images_file: Path, records: List[ImageRecord], new_paths: Dict[int, str], apply: bool) -> None:
    output = []
    for rec in records:
        output.append(f"{rec.image_id} {new_paths[rec.image_id]}\n")

    if apply:
        images_file.write_text("".join(output), encoding="utf-8")


def rename_tree(
    root: Path,
    class_map: Dict[str, str],
    apply: bool,
    progress_every: int,
) -> Tuple[int, int]:
    moved_files = 0
    moved_dirs = 0

    for idx, (old_class, new_class) in enumerate(class_map.items(), start=1):
        src_dir = root / old_class
        dst_dir = root / new_class

        if not src_dir.exists():
            continue

        files = [p for p in src_dir.iterdir() if p.is_file()]
        for file_path in files:
            new_name = stem_with_new_label(file_path.stem, old_class, new_class) + file_path.suffix
            target = dst_dir / new_name

            if apply:
                dst_dir.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Target already exists: {target}")
                shutil.move(str(file_path), str(target))

            moved_files += 1

        if apply and src_dir.exists():
            try:
                src_dir.rmdir()
                moved_dirs += 1
            except OSError:
                LOGGER.warning("Directory not empty after move: %s", src_dir)

        if idx % progress_every == 0:
            LOGGER.info("Processed %d class folders in %s", idx, root)

    return moved_files, moved_dirs


def write_mapping_report(class_map: Dict[str, str], out_csv: Path, apply: bool) -> None:
    if not apply:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_label", "new_label"])
        for old_label, new_label in class_map.items():
            writer.writerow([old_label, new_label])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean CUB class labels and rename image/segmentation folders and files. "
            "Updates classes.txt and images.txt consistently."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to CUB data directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply filesystem and metadata changes. Without this flag, script runs in dry-run mode.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Log progress after this many class folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    data_dir = args.data_dir
    classes_file = data_dir / "classes.txt"
    images_file = data_dir / "images.txt"
    images_root = data_dir / "images"
    segmentations_root = data_dir / "segmentations"
    report_file = data_dir / "label_mapping.csv"

    classes = read_classes(classes_file)
    class_map = make_class_mapping(classes)
    image_records = read_images(images_file)
    new_image_paths = build_new_image_paths(image_records, class_map)

    LOGGER.info("Total classes: %d", len(class_map))
    LOGGER.info("Total images in metadata: %d", len(image_records))
    LOGGER.info("Mode: %s", "APPLY" if args.apply else "DRY-RUN")

    write_classes(classes_file, classes, class_map, apply=args.apply)
    write_images(images_file, image_records, new_image_paths, apply=args.apply)

    moved_image_files, moved_image_dirs = rename_tree(
        root=images_root,
        class_map=class_map,
        apply=args.apply,
        progress_every=max(args.progress_every, 1),
    )
    moved_seg_files, moved_seg_dirs = rename_tree(
        root=segmentations_root,
        class_map=class_map,
        apply=args.apply,
        progress_every=max(args.progress_every, 1),
    )

    write_mapping_report(class_map, report_file, apply=args.apply)

    print("Preprocessing summary")
    print(f"- Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"- Classes mapped: {len(class_map)}")
    print(f"- Metadata entries: {len(image_records)}")
    print(f"- Image files to move/moved: {moved_image_files}")
    print(f"- Segmentation files to move/moved: {moved_seg_files}")
    print(f"- Image class folders removed: {moved_image_dirs}")
    print(f"- Segmentation class folders removed: {moved_seg_dirs}")
    print("- Parts files: unchanged (ID-based)")
    print("- Attribute files: unchanged (ID/order-based)")
    if args.apply:
        print(f"- Mapping report: {report_file}")


if __name__ == "__main__":
    main()
