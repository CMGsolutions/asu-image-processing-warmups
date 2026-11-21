"""TL;DR: Train binary/multiclass U-Nets and extract boxes from ground-truth masks.

Inputs: Segmentation01.zip and Segmentation02.zip with paired images and masks.
Outputs: Dice curves, qualitative figures, run log, and mask-derived bounding-box CSV.

Run with --help for configurable directories. Original experiment defaults are
preserved. Full training requires the datasets and has not been rerun here.
Related exercises share one script to retain their original dependencies.
"""

import argparse
from pathlib import Path


def main():
    """Parse locations before importing training libraries or touching datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing required dataset ZIP archives",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Destination for generated results; matching names are overwritten",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Dedicated scratch folder for extracted data",
    )
    args = parser.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_dir = (
        (args.work_dir or args.output_dir / "work" / Path(__file__).stem)
        .expanduser()
        .resolve()
    )
    if (
        args.work_dir == args.data_dir
        or args.work_dir == args.output_dir
        or args.work_dir in args.data_dir.parents
        or args.work_dir in args.output_dir.parents
    ):
        parser.error(
            "Choose a dedicated scratch directory separate from the input/output roots."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Save figures without requiring a desktop window. DataLoader uses a single
    # process so script-local dataset classes also work on Windows and macOS.
    import matplotlib

    matplotlib.use("Agg")

    # Locate segmentation archives and initialize outputs.
    from datetime import datetime
    import zipfile

    DRIVE_ROOT = args.data_dir
    DATA_DIR = args.data_dir
    WORK_DIR = args.work_dir
    OUTPUT_DIR = args.output_dir / "exercise_5_7_output"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def find_archive(token):
        """Locate the first matching dataset ZIP below the configured input directory."""
        matches = [
            p for p in DRIVE_ROOT.rglob("*.zip") if token.lower() in p.name.lower()
        ]
        if not matches:
            raise FileNotFoundError(
                f"Could not find a ZIP containing {token} in input directory."
            )
        return matches[0]

    ARCHIVES = {
        "exercise5": find_archive("Segmentation01"),
        "exercise6_7": find_archive("Segmentation02"),
    }
    DATASETS = {}
    for name, archive in ARCHIVES.items():
        target = WORK_DIR / name
        if not target.exists():
            target.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        DATASETS[name] = target
        print(f"Ready: {archive.name}")

    LOG_FILE = OUTPUT_DIR / "warmup_exercises_5_7_log.txt"
    with open(LOG_FILE, "w") as f:
        f.write(
            f"Warm-up Exercises 5-7 log\nStarted: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )

    def log(message=""):
        """Print an experiment message and persist it in the current run log."""
        print(message)
        with open(LOG_FILE, "a") as f:
            f.write(str(message) + "\n")

    log(f"Archives: {ARCHIVES}")

    # Match stems after removing _label suffixes; define U-Net and overlap scoring.
    import random, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader

    SEED, IMAGE_SIZE = 507, 256
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {DEVICE}; PyTorch: {torch.__version__}; seed: {SEED}")

    def find_dir(root, name):
        """Locate the dataset subdirectory or fail with its expected name."""
        choices = [p for p in Path(root).rglob(name) if p.is_dir()]
        if not choices:
            raise FileNotFoundError(f"Missing directory {name} below {root}")
        return choices[0]

    def load_gray(path):
        """Read an image as a single grayscale NumPy array."""
        return np.asarray(Image.open(path).convert("L"))

    def get_pairs(root):
        """Pair image and mask filenames across both supplied archive layouts."""

        def pair(image_dir, label_dir):
            """Match normalized stems; image extensions may differ from mask extensions."""
            valid = {".png", ".jpg", ".jpeg", ".bmp"}

            # Segmentation02 label files add the suffix _label (for example, case001_label.png).
            # Normalize names so they pair with case001.bmp correctly.
            def sample_key(path):
                """Remove the mask-only _label suffix before matching case IDs."""
                stem = path.stem.lower()
                return stem[:-6] if stem.endswith("_label") else stem

            images = {
                sample_key(p): p
                for p in image_dir.rglob("*")
                if p.suffix.lower() in valid
            }
            labels = {
                sample_key(p): p
                for p in label_dir.rglob("*")
                if p.suffix.lower() in valid
            }
            names = sorted(images.keys() & labels.keys())
            if not names:
                raise RuntimeError("No image/mask filename pairs found.")
            return [(images[n], labels[n]) for n in names]

        # Segmentation01 uses train/org and train/label; Segmentation02 uses org_train and label_train.
        try:
            train = pair(find_dir(root, "org_train"), find_dir(root, "label_train"))
            test = pair(find_dir(root, "org_test"), find_dir(root, "label_test"))
        except FileNotFoundError:
            train_root, test_root = find_dir(root, "train"), find_dir(root, "test")
            train = pair(find_dir(train_root, "org"), find_dir(train_root, "label"))
            test = pair(find_dir(test_root, "org"), find_dir(test_root, "label"))
        return train, test

    class UNet(nn.Module):
        def __init__(self, outputs):
            """Initialize the dataset index or model layers using the supplied settings."""
            super().__init__()

            def block(a, b):
                """Two padded convolutions preserve spatial size before downsampling."""
                return nn.Sequential(
                    nn.Conv2d(a, b, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(b, b, 3, padding=1),
                    nn.ReLU(),
                )

            self.e1, self.e2, self.e3 = block(1, 32), block(32, 64), block(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
            self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)
            self.head = nn.Conv2d(32, outputs, 1)

        def forward(self, x):
            """Apply the model and return its prediction (plus latent features for autoencoders)."""
            a = self.e1(x)
            b = self.e2(self.pool(a))
            c = self.e3(self.pool(b))
            b = self.d2(torch.cat([self.u2(c), b], 1))
            a = self.d1(torch.cat([self.u1(b), a], 1))
            return self.head(a)

    def dice(pred, truth, foreground_only=False):
        """Average overlap across present classes, optionally excluding background. Binary source scoring includes background; empty comparisons return one."""
        classes = sorted(set(np.unique(pred)) | set(np.unique(truth)))
        if foreground_only:
            classes = [c for c in classes if c != 0]
        scores = []
        for c in classes:
            p, t = pred == c, truth == c
            if p.sum() + t.sum():
                scores.append(2 * (p & t).sum() / (p.sum() + t.sum()))
        return float(np.mean(scores)) if scores else 1.0

    # Exercise 5: predict a binary lung mask using BCE logits. Source Dice averages present foreground/background classes.
    e5_train_pairs, e5_test_pairs = get_pairs(DATASETS["exercise5"])

    class BinaryDataset(Dataset):
        def __init__(self, pairs):
            """Initialize the dataset index or model layers using the supplied settings."""
            self.pairs = pairs

        def __len__(self):
            """Return the number of indexed samples."""
            return len(self.pairs)

        def __getitem__(self, i):
            """Load and transform one indexed sample for the training data loader."""
            ip, mp = self.pairs[i]
            image = (
                cv2.resize(
                    load_gray(ip),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                / 255.0
            )
            mask = (
                cv2.resize(
                    load_gray(mp),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                == 255
            )
            return torch.tensor(image[None], dtype=torch.float32), torch.tensor(
                mask[None], dtype=torch.float32
            )

    e5_train, e5_test = BinaryDataset(e5_train_pairs), BinaryDataset(e5_test_pairs)
    train_dl = DataLoader(e5_train, batch_size=8, shuffle=True, num_workers=0)
    test_dl = DataLoader(e5_test, batch_size=8, num_workers=0)
    E5_EPOCHS = 20
    model = UNet(1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    e5_scores = []
    log("\nEXERCISE 5: SEMANTIC LUNG SEGMENTATION")
    log(
        f"Train/test: {len(e5_train)}/{len(e5_test)}; epochs: {E5_EPOCHS}; lung label: 255"
    )
    for epoch in range(E5_EPOCHS):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in test_dl:
                p = (
                    (torch.sigmoid(model(x.to(DEVICE))) > 0.5)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                t = y.numpy().astype(np.uint8)
                scores += [dice(pp[0], tt[0]) for pp, tt in zip(p, t)]
        e5_scores.append(float(np.mean(scores)))
        log(
            f"Epoch {epoch + 1}: BCE={np.mean(losses):.4f}; test Dice={e5_scores[-1]:.3f}"
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, E5_EPOCHS + 1), e5_scores, marker="o")
    ax.set(
        xlabel="Epoch",
        ylabel="Test Dice",
        title="Exercise 5: semantic lung segmentation",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "exercise_5_dice.png"
    fig.savefig(path, dpi=300)
    plt.show()
    log(f"Curve: {path}")
    x, y = e5_test[0]
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x[None].to(DEVICE)))[0, 0].cpu().numpy() > 0.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [x[0], y[0], p], ["Chest X-ray", "Ground truth", "Prediction"]
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "exercise_5_example.png"
    fig.savefig(path, dpi=300)
    plt.show()
    log(f"Example: {path}")

    # Exercise 6: predict semantic classes, despite the original instance-segmentation title. Foreground Dice excludes background.
    e6_train_pairs, e6_test_pairs = get_pairs(DATASETS["exercise6_7"])
    LABEL_VALUES = sorted(
        {int(v) for _, mask in e6_train_pairs for v in np.unique(load_gray(mask))}
    )
    VALUE_TO_CLASS = {v: i for i, v in enumerate(LABEL_VALUES)}
    log("\nEXERCISE 6: INSTANCE ORGAN SEGMENTATION")
    log(
        f"Train/test: {len(e6_train_pairs)}/{len(e6_test_pairs)}; raw labels: {LABEL_VALUES}"
    )
    print("Raw mask values:", LABEL_VALUES)

    class MultiDataset(Dataset):
        def __init__(self, pairs):
            """Initialize the dataset index or model layers using the supplied settings."""
            self.pairs = pairs

        def __len__(self):
            """Return the number of indexed samples."""
            return len(self.pairs)

        def __getitem__(self, i):
            """Load and transform one indexed sample for the training data loader."""
            ip, mp = self.pairs[i]
            image = (
                cv2.resize(
                    load_gray(ip),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                / 255.0
            )
            raw = cv2.resize(
                load_gray(mp), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
            )
            encoded = np.zeros_like(raw, dtype=np.int64)
            for value, class_id in VALUE_TO_CLASS.items():
                encoded[raw == value] = class_id
            return torch.tensor(image[None], dtype=torch.float32), torch.tensor(
                encoded, dtype=torch.long
            )

    e6_train, e6_test = MultiDataset(e6_train_pairs), MultiDataset(e6_test_pairs)
    train_dl = DataLoader(e6_train, batch_size=8, shuffle=True, num_workers=0)
    test_dl = DataLoader(e6_test, batch_size=8, num_workers=0)
    E6_EPOCHS = 25
    model6 = UNet(len(LABEL_VALUES)).to(DEVICE)
    opt6 = torch.optim.Adam(model6.parameters(), lr=1e-3)
    loss6 = nn.CrossEntropyLoss()
    e6_scores = []
    for epoch in range(E6_EPOCHS):
        model6.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt6.zero_grad()
            loss = loss6(model6(x), y)
            loss.backward()
            opt6.step()
            losses.append(loss.item())
        model6.eval()
        scores = []
        with torch.no_grad():
            for x, y in test_dl:
                p = model6(x.to(DEVICE)).argmax(1).cpu().numpy()
                t = y.numpy()
                scores += [dice(pp, tt, foreground_only=True) for pp, tt in zip(p, t)]
        e6_scores.append(float(np.mean(scores)))
        log(
            f"Epoch {epoch + 1}: CE={np.mean(losses):.4f}; foreground Dice={e6_scores[-1]:.3f}"
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, E6_EPOCHS + 1), e6_scores, marker="o", color="#d62728")
    ax.set(
        xlabel="Epoch",
        ylabel="Foreground Dice",
        title="Exercise 6: instance organ segmentation",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "exercise_6_dice.png"
    fig.savefig(path, dpi=300)
    plt.show()
    log(f"Curve: {path}")
    x, y = e6_test[0]
    model6.eval()
    with torch.no_grad():
        p = model6(x[None].to(DEVICE)).argmax(1)[0].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [x[0], y, p], ["Chest X-ray", "Ground truth", "Prediction"]
    ):
        ax.imshow(img, cmap="tab20")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "exercise_6_example.png"
    fig.savefig(path, dpi=300)
    plt.show()
    log(f"Example: {path}")

    # Exercise 7: extract connected components from supplied masks, not predictions. This is annotation conversion, not detector evaluation.
    def make_boxes(image_path, mask_path):
        """Convert each nonzero mask label into 8-connected-component boxes. Reject tiny components and components occupying over 75% of the image."""
        mask = load_gray(mask_path)
        height, width = mask.shape
        output = []
        for value in np.unique(mask):
            if value == 0:
                continue
            count, _, stats, centroids = cv2.connectedComponentsWithStats(
                (mask == value).astype(np.uint8), 8
            )
            for n in range(1, count):
                x, y, w, h, area = stats[n]
                if area < 25 or area > 0.75 * height * width:
                    continue
                output.append(
                    {
                        "image": image_path.name,
                        "mask_value": int(value),
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area),
                        "center_x": round(float(centroids[n][0]), 2),
                        "center_y": round(float(centroids[n][1]), 2),
                    }
                )
        return output

    records = [record for ip, mp in e6_test_pairs for record in make_boxes(ip, mp)]
    boxes = pd.DataFrame(records).sort_values(
        ["image", "mask_value", "area"], ascending=[True, True, False]
    )
    csv_path = OUTPUT_DIR / "exercise_7_test_bounding_boxes.csv"
    boxes.to_csv(csv_path, index=False)
    log("\nEXERCISE 7: ORGAN LOCALIZATION")
    log(f"Created {len(boxes)} boxes from {len(e6_test_pairs)} test masks.")
    log(f"Bounding-box CSV: {csv_path}")
    print(boxes.head(12).to_string(index=False))
    ip, mp = e6_test_pairs[0]
    image = load_gray(ip)
    example = make_boxes(ip, mp)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(example))))
    for color, box in zip(colors, example):
        ax.add_patch(
            plt.Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        ax.text(
            box["x"],
            max(0, box["y"] - 3),
            f"label {box['mask_value']}",
            color=color,
            fontsize=8,
            weight="bold",
        )
    ax.set_title("Exercise 7: boxes derived from Segmentation02")
    ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "exercise_7_localization_example.png"
    fig.savefig(path, dpi=300)
    plt.show()
    log(f"Visualization: {path}")
    log(f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
