"""TL;DR: Train a pretrained ResNet-18 orientation classifier with a replacement head.

Inputs: Directions01.zip containing train/test class folders.
Outputs: Per-epoch metrics CSV, accuracy plot, and final classifier weights.

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

    # Resolve the orientation dataset and output locations.

    DRIVE_FOLDER = args.data_dir
    DIRECTIONS_ZIP = DRIVE_FOLDER / "Directions01.zip"
    WORK_DIR = args.work_dir
    OUTPUT_DIR = args.output_dir / "exercise_2_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    assert DIRECTIONS_ZIP.exists(), f"Missing: {DIRECTIONS_ZIP}"
    print("Directions01.zip found.")

    # Extract and locate labeled training and held-out folders.
    import shutil

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    shutil.unpack_archive(DIRECTIONS_ZIP, WORK_DIR)

    train_dirs = [
        path
        for path in WORK_DIR.rglob("*")
        if path.is_dir() and path.name.lower() == "train"
    ]
    test_dirs = [
        path
        for path in WORK_DIR.rglob("*")
        if path.is_dir() and path.name.lower() in {"test", "eval", "evaluation"}
    ]
    assert train_dirs, "Could not find the training folder."
    assert test_dirs, "Could not find the test folder."
    TRAIN_DIR, TEST_DIR = train_dirs[0], test_dirs[0]
    print("Training folder:", TRAIN_DIR)
    print("Test folder:", TEST_DIR)

    # Use ImageNet scaling and three grayscale channels for pretrained weights.
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CPU is supported, although training may be substantially slower.
    print("Using device:", device)

    # Convert grayscale X-rays to three channels for pretrained ResNet18.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    print("Classes:", train_data.classes)
    print("Training images:", len(train_data), "Test images:", len(test_data))

    # Freeze pretrained parameters and optimize only the new classification head.
    from torchvision.models import ResNet18_Weights

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Fine-tune only the classifier first. This is appropriate for the small training set.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    NUM_EPOCHS = 10
    print(model.fc)

    # Train with cross-entropy; evaluate without gradient tracking each epoch.
    history = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        correct = total = 0
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / len(train_loader),
            "train_accuracy": correct / total,
            "test_accuracy": test_correct / test_total,
        }
        history.append(record)
        print(
            f"Epoch {record['epoch']}/{NUM_EPOCHS}: loss={record['loss']:.4f}, train={record['train_accuracy']:.2%}, test={record['test_accuracy']:.2%}"
        )

    # Persist metrics, plot, and final state dictionary.
    import csv
    import matplotlib.pyplot as plt

    with open(OUTPUT_DIR / "orientation_metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(7, 4))
    plt.plot(
        [row["epoch"] for row in history],
        [row["train_accuracy"] for row in history],
        marker="o",
        label="Training",
    )
    plt.plot(
        [row["epoch"] for row in history],
        [row["test_accuracy"] for row in history],
        marker="o",
        label="Test",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "orientation_accuracy.png", dpi=150)
    plt.show()

    torch.save(model.state_dict(), OUTPUT_DIR / "resnet18_orientation.pt")
    print("Saved metrics, plot, and model to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
