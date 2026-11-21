"""TL;DR: Train a grayscale ResNet-18 from scratch on MNIST; source has no warmup number.

Inputs: MNIST, downloaded automatically into the selected work directory.
Outputs: Printed epoch losses and final test accuracy; no checkpoint in the source workflow.

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

    # Choose CUDA when available and seed initialization.
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models
    from torchvision.transforms import ToTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 100
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    torch.manual_seed(42)

    # Download MNIST with ToTensor only, preserving the original input pipeline.
    # Same data conversion as the original CNN: no normalization or augmentation.
    train_data = datasets.MNIST(
        root=str(args.work_dir / "mnist"),
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root=str(args.work_dir / "mnist"),
        train=False,
        transform=ToTensor(),
        download=True,
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    print(train_data.data.size(), test_data.data.size())

    # Adapt the first convolution to 1-channel 28x28 digits and remove max pooling.
    # Standard ResNet18 residual stages, adapted only for grayscale 28x28 inputs.
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    # Train the complete model from scratch with cross-entropy and Adam.
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}: loss = {total_loss / len(train_loader):.4f}"
        )

    # Measure held-out accuracy without updating weights.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"ResNet18 test accuracy: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
