"""Train the submitted Day 1 YOLO26m-cls classifier in two phases."""
import argparse
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

from infer import CLASSES, EXTENSIONS, IMAGE_SIZE, prepare_image

SEED = 42
BATCH_SIZE = 7
PATTERNS = ("grid", "circle")



def open_rgb(path):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    return image


def augment_image(image, rng):
    w, h = image.size
    dx, dy = round(w * 0.03), round(h * 0.05)
    pixels = np.asarray(image.crop((dx, dy, w - dx, h - dy)))
    h, w = pixels.shape[:2]
    turned = ndimage.rotate(
        pixels, rng.uniform(-20, 20), axes=(0, 1), reshape=False,
        order=1, mode="reflect",
    )
    moved = ndimage.shift(
        turned, (rng.uniform(-0.06, 0.06) * h, rng.uniform(-0.08, 0.08) * w, 0),
        order=1, mode="reflect",
    )
    augmented = Image.fromarray(moved)
    if rng.random() < 0.5:
        augmented = ImageOps.mirror(augmented)
    return ImageEnhance.Brightness(augmented).enhance(rng.uniform(0.92, 1.08))


def cut_beans(image):
    gray = np.asarray(image.convert("L"), dtype=np.int16)
    paper = ndimage.grey_closing(gray, size=(55, 55))
    foreground = ndimage.binary_opening(paper - gray >= 26, structure=np.ones((3, 3)))

    labels, _ = ndimage.label(foreground)
    areas = np.bincount(labels.ravel())
    h, w = gray.shape
    cutouts = []

    for idx, (y, x) in enumerate(ndimage.find_objects(labels), start=1):
        if not (55 <= areas[idx] <= 2700):
            continue
        if min(y.start, x.start, h - y.stop, w - x.stop) < 5:
            continue

        x0, y0, x1, y1 = x.start - 4, y.start - 4, x.stop + 4, y.stop + 4
        alpha = labels[y0:y1, x0:x1] == idx
        alpha = ndimage.binary_dilation(alpha, iterations=2)
        mask = Image.fromarray(255 * alpha.astype(np.uint8))

        bean = image.crop((x0, y0, x1, y1)).convert("RGBA")
        bean.putalpha(mask.filter(ImageFilter.GaussianBlur(0.6)))
        cutouts.append(bean)

    return cutouts


def make_pile(pool, backgrounds, rng):
    canvas = rng.choice(backgrounds).copy()
    w, h = canvas.size
    occupied = np.zeros((h, w), dtype=bool)

    center_x = rng.uniform(0.30, 0.70) * w
    center_y = rng.uniform(0.30, 0.70) * h
    spread = rng.uniform(52, 88)
    max_overlap = rng.choice([0.02, 0.10, 0.20])
    count = min(rng.randint(7, 9), len(pool))
    placed = 0

    for bean in rng.sample(pool, count):
        scale = rng.uniform(0.9, 1.1)
        size = (round(bean.width * scale), round(bean.height * scale))
        patch = bean.resize(size, Image.Resampling.BICUBIC)
        patch = patch.rotate(rng.uniform(0, 360), expand=True, resample=Image.Resampling.BICUBIC)

        alpha = patch.getchannel("A")
        visible = np.asarray(alpha) > 127
        bw, bh = patch.size
        for _ in range(80):
            x = round(center_x + rng.uniform(-spread, spread) - bw / 2)
            y = round(center_y + rng.uniform(-spread, spread) - bh / 2)
            if x < 0 or y < 0 or x + bw > w or y + bh > h:
                continue

            region = occupied[y:y + bh, x:x + bw]
            overlap = (region & visible).sum() / visible.sum()
            if overlap <= max_overlap:
                canvas.paste(patch, (x, y), alpha)
                region |= visible
                placed += 1
                break

    return canvas, placed


def prepare_raw_dataset(source, target):
    """Generate the training split and its offline augmentation from Day 1 photos."""
    rng = random.Random(SEED)
    training_photos = {}
    for label in CLASSES:
        class_dir = target / label
        class_dir.mkdir(parents=True, exist_ok=True)
        training_photos[label] = [source / label / f"{pattern}_{label}.jpg" for pattern in PATTERNS]
        for photo in training_photos[label]:
            shutil.copy2(photo, class_dir / photo.name)
            image = open_rgb(photo)
            for i in range(5):
                augmented = augment_image(image, rng)
                augmented.save(class_dir / f"aug_{photo.stem}_{i:02d}.jpg", quality=95)

    bean_pool = {label: [] for label in CLASSES}
    backgrounds = []
    for label, photos in training_photos.items():
        for photo in photos:
            image = open_rgb(photo)
            bean_pool[label].extend(cut_beans(image))
            paper = ndimage.grey_closing(np.asarray(image), size=(65, 65, 1))
            backgrounds.append(Image.fromarray(paper).filter(ImageFilter.GaussianBlur(9)))

    for label, pool in bean_pool.items():
        if len(pool) < 7:
            raise ValueError(f"Not enough bean cutouts for {label}")
        for i in range(10):
            for _ in range(10):
                pile, placed = make_pile(pool, backgrounds, rng)
                if placed >= 6:
                    break
            else:
                raise RuntimeError(f"Could not synthesize a pile for {label}")
            pile.save(target / label / f"paste_{label}_{i:02d}.jpg", quality=95)
    return target


def prepare_fit_directory(source, fit):
    train = fit / "train"
    for label in CLASSES:
        images = sorted(p for p in (source / label).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f"No training images for {label}")
        class_dir = train / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            prepare_image(image, IMAGE_SIZE).save(class_dir / f"{image.stem}.png")
    # Ultralytics may run an internal validation pass at the final epoch.
    # It uses only the training data here. The square holdout is outside fit.
    (fit / "val").symlink_to(train, target_is_directory=True)


def train_classifier(fit, run_dir, device):
    from ultralytics import YOLO

    common = dict(
        data=str(fit), imgsz=IMAGE_SIZE, batch=BATCH_SIZE, nbs=BATCH_SIZE,
        device=device, workers=0, seed=SEED, optimizer="AdamW",
        warmup_epochs=0, val=False, patience=0, scale=0.0, fliplr=0.0,
        flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, auto_augment=None,
        erasing=0.0, project=str(run_dir), plots=False,
    )
    model = YOLO("yolo26m-cls.pt")
    head_index = len(model.model.model) - 1
    model.train(**common, epochs=5, freeze=head_index, lr0=0.0005, name="yolo26m_freeze_head")

    model = YOLO(str(model.trainer.last))
    model.train(**common, epochs=25, freeze=0, lr0=0.0001, name="yolo26m_unfreeze_all")
    return Path(model.trainer.last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted day_1 data or prepared dataset root")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index, e.g. 0")
    parser.add_argument("--out", type=Path, default=Path("weights_retrained"), help="Directory for the new checkpoint")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="day1-train-") as temp:
        work = Path(temp)
        training = args.data / "train"
        if not training.is_dir():
            training = prepare_raw_dataset(args.data, work / "augmented")
        fit = work / "fit"
        prepare_fit_directory(training, fit)
        last = train_classifier(fit, work / "runs", args.device)
        args.out.mkdir(parents=True, exist_ok=True)
        destination = args.out / "best.pt"
        shutil.copy2(last, destination)
        print(f"Checkpoint: {destination.resolve()}")


if __name__ == "__main__":
    main()
