"""TL;DR: Fine-tune ResNet-18 for supplied gender labels and age regression.

Inputs: Gender01.zip and XPAge01_RGB.zip; age CSVs need filenames and age columns.
Outputs: Experiment log, gender accuracy curve, and age MAE curve in years.

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

    # Set up local directories for both related experiments.

    DRIVE_ROOT = args.data_dir
    DATA_DIR = args.data_dir
    WORK_DIR = args.work_dir
    EX3_OUTPUT_DIR = args.output_dir / "exercise_3_output"
    EX4_OUTPUT_DIR = args.output_dir / "exercise_4_output"
    for folder in (WORK_DIR, EX3_OUTPUT_DIR, EX4_OUTPUT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    print("Input archive folder:", DATA_DIR)
    print("Exists:", DATA_DIR.exists())

    # Find archives and initialize the persistent experiment log.
    import os, platform, random, sys, zipfile
    from datetime import datetime

    ARCHIVES = {"exercise3_gender": "Gender01.zip", "exercise4_age": "XPAge01_RGB.zip"}
    zip_lookup = {p.name: p for p in DRIVE_ROOT.rglob("*.zip")}
    missing = [filename for filename in ARCHIVES.values() if filename not in zip_lookup]
    if missing:
        raise FileNotFoundError(f"Missing archives in input directory: {missing}")

    DATASETS = {}
    for name, filename in ARCHIVES.items():
        target = WORK_DIR / name
        if not target.exists():
            target.mkdir(parents=True)
            with zipfile.ZipFile(zip_lookup[filename]) as zf:
                zf.extractall(target)
        DATASETS[name] = target
        print(f"Ready: {filename} -> {target}")

    LOG_FILE = args.output_dir / "warmup_exercises_3_4_log.txt"
    with open(LOG_FILE, "w") as f:
        f.write(
            f"Warm-up Exercises 3 and 4 experiment log\nStarted: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )

    def log(message=""):
        """Print an experiment message and persist it in the current run log."""
        print(message)
        with open(LOG_FILE, "a") as f:
            f.write(str(message) + "\n")

    log(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    log(f"Archives: {ARCHIVES}")
    log(f"Output log: {LOG_FILE}")

    # Seed training and define the shared RGB/ImageNet input pipeline.
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models

    SEED = 507
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {DEVICE}; PyTorch: {torch.__version__}; seed: {SEED}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class FolderImageDataset(Dataset):
        def __init__(self, root, labels, transform):
            """Initialize the dataset index or model layers using the supplied settings."""
            index = {label.lower(): i for i, label in enumerate(labels)}
            self.samples = [
                (p, index[p.parent.name.lower()])
                for p in Path(root).rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                and p.parent.name.lower() in index
            ]
            if not self.samples:
                raise RuntimeError(f"No labeled images found under {root}")
            self.transform = transform

        def __len__(self):
            """Return the number of indexed samples."""
            return len(self.samples)

        def __getitem__(self, i):
            """Load and transform one indexed sample for the training data loader."""
            path, label = self.samples[i]
            return self.transform(Image.open(path).convert("RGB")), label

    # Exercise 3: fine-tune all ResNet parameters on the supplied binary labels.
    GENDER_LABELS = ["male", "female"]
    gender_root = next(
        p for p in DATASETS["exercise3_gender"].rglob("train") if p.is_dir()
    ).parent
    train_gender = FolderImageDataset(gender_root / "train", GENDER_LABELS, transform)
    test_gender = FolderImageDataset(gender_root / "test", GENDER_LABELS, transform)
    train_dl = DataLoader(
        train_gender, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(test_gender, batch_size=64, num_workers=0, pin_memory=True)

    E3_EPOCHS, E3_BATCH_SIZE, E3_LR = 8, 32, 3e-4
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(GENDER_LABELS))
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=E3_LR)
    loss_fn = nn.CrossEntropyLoss()

    log("\nEXERCISE 3: GENDER CLASSIFICATION")
    log(f"Dataset: Gender01 | classes: {GENDER_LABELS}")
    log(f"Train images: {len(train_gender)} | test images: {len(test_gender)}")
    log(
        f"Model: pretrained ResNet-18 | epochs: {E3_EPOCHS} | batch size: {E3_BATCH_SIZE} | optimizer: AdamW | learning rate: {E3_LR}"
    )

    e3_accuracy, e3_loss = [], []
    for epoch in range(E3_EPOCHS):
        model.train()
        batch_losses = []
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                prediction = model(x.to(DEVICE)).argmax(1).cpu()
                correct += (prediction == y).sum().item()
                total += len(y)
        accuracy = correct / total
        mean_loss = float(np.mean(batch_losses))
        e3_accuracy.append(accuracy)
        e3_loss.append(mean_loss)
        log(
            f"Epoch {epoch + 1}: train loss = {mean_loss:.4f}; test accuracy = {accuracy:.3f} ({correct}/{total})"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        range(1, E3_EPOCHS + 1), e3_accuracy, marker="o", color="#1f77b4", linewidth=2
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Test accuracy",
        title="Exercise 3: Gender classification accuracy",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    E3_FIGURE = EX3_OUTPUT_DIR / "exercise_3_accuracy.png"
    fig.savefig(E3_FIGURE, dpi=300, bbox_inches="tight")
    plt.show()
    log(f"Final test accuracy: {e3_accuracy[-1]:.3f}")
    log(f"Report figure saved: {E3_FIGURE}")

    # Exercise 4: normalize targets using training statistics only; undo scaling for MAE.
    age_root = DATASETS["exercise4_age"]
    csvs = list(age_root.rglob("*.csv"))
    train_csv = next(p for p in csvs if p.name.lower() == "trainingdata.csv")
    test_csv = next(p for p in csvs if p.name.lower() == "testdata.csv")
    train_table, test_table = pd.read_csv(train_csv), pd.read_csv(test_csv)
    FILE_COLUMN, AGE_COLUMN = "filenames", "age"

    class AgeDataset(Dataset):
        def __init__(self, root, table, age_mean, age_std):
            """Initialize the dataset index or model layers using the supplied settings."""
            files = {
                p.name: p
                for p in Path(root).rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            }
            self.samples = [
                (files[str(row[FILE_COLUMN])], float(row[AGE_COLUMN]))
                for _, row in table.iterrows()
                if str(row[FILE_COLUMN]) in files
            ]
            if not self.samples:
                raise RuntimeError("CSV image names did not match the image files.")
            self.age_mean, self.age_std = age_mean, age_std

        def __len__(self):
            """Return the number of indexed samples."""
            return len(self.samples)

        def __getitem__(self, i):
            """Load and transform one indexed sample for the training data loader."""
            path, age = self.samples[i]
            normalized_age = (age - self.age_mean) / self.age_std
            return transform(Image.open(path).convert("RGB")), torch.tensor(
                normalized_age, dtype=torch.float32
            )

    age_mean, age_std = train_table[AGE_COLUMN].mean(), train_table[AGE_COLUMN].std()
    train_age = AgeDataset(age_root, train_table, age_mean, age_std)
    test_age = AgeDataset(age_root, test_table, age_mean, age_std)
    train_dl = DataLoader(
        train_age, batch_size=16, shuffle=True, num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(test_age, batch_size=32, num_workers=0, pin_memory=True)

    E4_EPOCHS, E4_BATCH_SIZE, E4_LR = 60, 16, 1e-4
    age_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    age_model.fc = nn.Linear(age_model.fc.in_features, 1)
    age_model = age_model.to(DEVICE)
    age_optimizer = torch.optim.AdamW(
        age_model.parameters(), lr=E4_LR, weight_decay=1e-4
    )
    age_loss_fn = nn.MSELoss()

    log("\nEXERCISE 4: AGE ESTIMATION")
    log(
        f"Dataset: XPAge01_RGB | train CSV: {train_csv.name}; test CSV: {test_csv.name}"
    )
    log(
        f"Train images: {len(train_age)} | test images: {len(test_age)} | training age mean/std: {age_mean:.2f}/{age_std:.2f}"
    )
    log(
        f"Model: pretrained ResNet-18 regression head | epochs: {E4_EPOCHS} | batch size: {E4_BATCH_SIZE} | optimizer: AdamW | learning rate: {E4_LR}"
    )

    e4_mae = []
    for epoch in range(E4_EPOCHS):
        age_model.train()
        batch_losses = []
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            age_optimizer.zero_grad()
            loss = age_loss_fn(age_model(x).squeeze(1), y)
            loss.backward()
            age_optimizer.step()
            batch_losses.append(loss.item())
        age_model.eval()
        absolute_errors = []
        with torch.no_grad():
            for x, y in test_dl:
                predicted_age = (
                    age_model(x.to(DEVICE)).squeeze(1).cpu() * age_std + age_mean
                )
                true_age = y * age_std + age_mean
                absolute_errors.extend((predicted_age - true_age).abs().tolist())
        mae = float(np.mean(absolute_errors))
        e4_mae.append(mae)
        log(
            f"Epoch {epoch + 1}: train MSE = {np.mean(batch_losses):.4f}; test MAE = {mae:.2f} years"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, E4_EPOCHS + 1), e4_mae, color="#d62728", linewidth=2)
    ax.set(
        xlabel="Epoch",
        ylabel="Test MAE (years)",
        title="Exercise 4: Age-estimation error",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    E4_FIGURE = EX4_OUTPUT_DIR / "exercise_4_mae.png"
    fig.savefig(E4_FIGURE, dpi=300, bbox_inches="tight")
    plt.show()
    log(f"Final test MAE: {e4_mae[-1]:.2f} years")
    log(f"Exercise 4 figure saved: {E4_FIGURE}")
    log(f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
