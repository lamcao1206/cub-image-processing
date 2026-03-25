import random
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/CUB_200_2011")

TRAIN_TEST_SPLIT_FILE = DATA_DIR / "train_test_split.txt"
IMAGE_CLASS_LABELS_FILE = DATA_DIR / "image_class_labels.txt"

OUTPUT_FILE = DATA_DIR / "train_val_test_split.txt"

VAL_RATIO = 0.2
SEED = 42


def load_int_pairs(path: Path):
    mapping = {}
    with path.open("r") as f:
        for line in f:
            k, v = line.strip().split()
            mapping[int(k)] = int(v)
    return mapping


def main():
    random.seed(SEED)

    split_map = load_int_pairs(TRAIN_TEST_SPLIT_FILE)
    class_map = load_int_pairs(IMAGE_CLASS_LABELS_FILE)

    # group TRAIN images by class
    class_to_images = defaultdict(list)

    for image_id, is_train in split_map.items():
        if is_train == 1:
            class_id = class_map[image_id]
            class_to_images[class_id].append(image_id)

    train_ids = set()
    val_ids = set()
    test_ids = set()

    # split each class
    for class_id, image_ids in class_to_images.items():
        random.shuffle(image_ids)

        n_val = int(len(image_ids) * VAL_RATIO)

        val_subset = image_ids[:n_val]
        train_subset = image_ids[n_val:]

        val_ids.update(val_subset)
        train_ids.update(train_subset)

    # keep original test
    for image_id, is_train in split_map.items():
        if is_train == 0:
            test_ids.add(image_id)

    # write output
    with OUTPUT_FILE.open("w") as f:
        for image_id in sorted(split_map.keys()):
            if image_id in train_ids:
                split = "train"
            elif image_id in val_ids:
                split = "val"
            else:
                split = "test"

            f.write(f"{image_id} {split}\n")

    print("✅ Done!")
    print(f"Train: {len(train_ids)}")
    print(f"Val  : {len(val_ids)}")
    print(f"Test : {len(test_ids)}")


if __name__ == "__main__":
    main()
