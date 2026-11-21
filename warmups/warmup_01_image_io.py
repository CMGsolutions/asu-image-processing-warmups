"""TL;DR: Read, display, and write PNG, JPEG, and DICOM chest radiographs.

Inputs: Practice_PNGandJPG.zip and Practice_DICOM.zip.
Outputs: Image copies, DICOM copy, normalized PNG preview, and comparison figures.

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

    # Configure archive locations and output folders.

    DRIVE_FOLDER = args.data_dir
    PNG_JPG_ZIP = DRIVE_FOLDER / "Practice_PNGandJPG.zip"
    DICOM_ZIP = DRIVE_FOLDER / "Practice_DICOM.zip"

    WORK_DIR = args.work_dir
    PNG_JPG_DIR = WORK_DIR / "practice_png_jpg"
    DICOM_DIR = WORK_DIR / "practice_dicom"
    OUTPUT_DIR = args.output_dir / "exercise_1_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    assert PNG_JPG_ZIP.exists(), f"Missing: {PNG_JPG_ZIP}"
    assert DICOM_ZIP.exists(), f"Missing: {DICOM_ZIP}"
    print("ZIP files found.")

    # Extract the two archives into the dedicated scratch directory.
    import shutil

    if PNG_JPG_DIR.exists():
        shutil.rmtree(PNG_JPG_DIR)
    if DICOM_DIR.exists():
        shutil.rmtree(DICOM_DIR)

    shutil.unpack_archive(PNG_JPG_ZIP, PNG_JPG_DIR)
    shutil.unpack_archive(DICOM_ZIP, DICOM_DIR)
    print(f"Extracted files into {WORK_DIR}")

    # Inventory files before selecting representative inputs.
    from collections import Counter

    png_jpg_files = [path for path in PNG_JPG_DIR.rglob("*") if path.is_file()]
    dicom_files = [path for path in DICOM_DIR.rglob("*") if path.is_file()]

    print(
        "PNG/JPG files:",
        len(png_jpg_files),
        Counter(path.suffix.lower() for path in png_jpg_files),
    )
    print(
        "DICOM files:",
        len(dicom_files),
        Counter(path.suffix.lower() or "[no extension]" for path in dicom_files),
    )
    print("Example PNG/JPG file:", png_jpg_files[0])
    print("Example DICOM file:", dicom_files[0])

    # Read grayscale PNG/JPEG pixels; preserve the source format when saving.
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    png_file = next(path for path in png_jpg_files if path.suffix.lower() == ".png")
    jpg_file = next(
        path for path in png_jpg_files if path.suffix.lower() in {".jpg", ".jpeg"}
    )

    png_image = Image.open(png_file).convert("L")
    jpg_image = Image.open(jpg_file).convert("L")

    for name, image in [("PNG", png_image), ("JPG", jpg_image)]:
        pixels = np.asarray(image)
        print(
            f"{name}: size={image.size}, mode={image.mode}, pixel range={pixels.min()} to {pixels.max()}"
        )

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(png_image, cmap="gray")
    axes[0].set_title("PNG chest X-ray")
    axes[1].imshow(jpg_image, cmap="gray")
    axes[1].set_title("JPG chest X-ray")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "png_jpg_display.png", dpi=150)
    plt.show()

    png_image.save(OUTPUT_DIR / "written_from_png.png")
    jpg_image.save(OUTPUT_DIR / "written_from_jpg.jpg", quality=95)
    print("Wrote PNG and JPG copies to:", OUTPUT_DIR)

    # Read DICOM pixels plus metadata; preview rescaling is for display only.
    import pydicom

    dicom_file = dicom_files[0]
    dataset = pydicom.dcmread(dicom_file)
    pixels = dataset.pixel_array

    print("DICOM file:", dicom_file.name)
    print("Modality:", getattr(dataset, "Modality", "not recorded"))
    print("Rows x Columns:", dataset.Rows, "x", dataset.Columns)
    print("Bits stored:", getattr(dataset, "BitsStored", "not recorded"))
    print("Pixel range:", pixels.min(), "to", pixels.max())

    plt.figure(figsize=(6, 6))
    plt.imshow(pixels, cmap="gray")
    plt.title("DICOM chest X-ray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dicom_display.png", dpi=150)
    plt.show()

    # Write a DICOM copy and an 8-bit PNG display copy.
    dataset.save_as(OUTPUT_DIR / "written_dicom_copy.dcm")
    # Avoid division by zero for a constant-intensity image.
    span = float(pixels.max()) - float(pixels.min())
    scaled = ((pixels.astype(np.float64) - pixels.min()) / max(span, 1.0) * 255).astype(
        np.uint8
    )
    Image.fromarray(scaled).save(OUTPUT_DIR / "dicom_as_png.png")
    print("Wrote DICOM and PNG output files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
