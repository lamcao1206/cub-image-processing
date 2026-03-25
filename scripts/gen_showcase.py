"""
Generate a showcase of random CUB-200-2011 images with bounding boxes drawn.

Picks 10 random images from the dataset, copies the originals into
  showcase/original/
and saves versions with the bounding box rectangle drawn into
  showcase/hbox/
"""

from pathlib import Path
import logging
import random
import shutil

from PIL import Image, ImageDraw, ImageFont

# ── paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "CUB_200_2011"

IMAGES_FILE       = DATA_DIR / "images.txt"
BOUNDING_BOX_FILE = DATA_DIR / "bounding_boxes.txt"
IMAGES_DIR        = DATA_DIR / "images"

SHOWCASE_DIR  = PROJECT_ROOT / "showcase"
ORIGINAL_DIR  = SHOWCASE_DIR / "original"
HBOX_DIR      = SHOWCASE_DIR / "hbox"

NUM_SAMPLES   = 10
BOX_COLOR     = (0, 255, 0)   # green
BOX_WIDTH     = 3
LABEL_COLOR   = (255, 255, 0) # yellow

LOGGER = logging.getLogger("gen_showcase")


# ── helpers ──────────────────────────────────────────────────────────────
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_images_map(path: Path) -> dict[int, str]:
    """Return {image_id: relative_path} from images.txt."""
    mapping: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def load_bounding_boxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
    """Return {image_id: (x, y, w, h)} from bounding_boxes.txt."""
    boxes: dict[int, tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                img_id = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes[img_id] = (x, y, w, h)
    return boxes


def draw_bounding_box(
    img: Image.Image,
    bbox: tuple[float, float, float, float],
    label: str,
) -> Image.Image:
    """Draw a rectangle + label on *a copy* of the image and return it."""
    result = img.copy()
    draw = ImageDraw.Draw(result)

    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # rectangle
    draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)

    # label background + text
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    text = label
    text_bbox = draw.textbbox((0, 0), text, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]

    # place label just above the box; fall back to inside if near top edge
    label_y = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 2
    draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=BOX_COLOR)
    draw.text((x1 + 3, label_y + 2), text, fill=(0, 0, 0), font=font)

    # bbox dimensions text at bottom
    dim_text = f"bbox: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})"
    draw.text((x1, y2 + 4), dim_text, fill=LABEL_COLOR, font=font)

    return result


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    configure_logging()

    # load metadata
    images_map = load_images_map(IMAGES_FILE)
    bbox_map   = load_bounding_boxes(BOUNDING_BOX_FILE)

    LOGGER.info("Loaded %d images and %d bounding boxes", len(images_map), len(bbox_map))

    # pick random samples
    all_ids = sorted(images_map.keys())
    sample_ids = random.sample(all_ids, min(NUM_SAMPLES, len(all_ids)))
    LOGGER.info("Selected %d random images: %s", len(sample_ids), sample_ids)

    # prepare output dirs
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    HBOX_DIR.mkdir(parents=True, exist_ok=True)

    for img_id in sample_ids:
        rel_path = images_map[img_id]
        src_path = IMAGES_DIR / rel_path

        if not src_path.exists():
            LOGGER.warning("Image not found, skipping: %s", src_path)
            continue

        # derive a readable filename: <id>_<breed>_<filename>
        breed = rel_path.split("/")[0] if "/" in rel_path else "unknown"
        filename = src_path.name
        output_name = f"{img_id:05d}_{breed}_{filename}"

        # ── copy original ────────────────────────────────────────────
        dst_original = ORIGINAL_DIR / output_name
        shutil.copy2(src_path, dst_original)
        LOGGER.info("Copied original: %s", dst_original.name)

        # ── draw bbox and save ───────────────────────────────────────
        bbox = bbox_map.get(img_id)
        if bbox is None:
            LOGGER.warning("No bounding box for image %d, skipping hbox", img_id)
            continue

        img = Image.open(src_path).convert("RGB")
        annotated = draw_bounding_box(img, bbox, label=breed)
        dst_hbox = HBOX_DIR / output_name
        annotated.save(dst_hbox)
        LOGGER.info("Saved hbox image: %s", dst_hbox.name)

    LOGGER.info("Done! Check %s", SHOWCASE_DIR)


if __name__ == "__main__":
    main()
