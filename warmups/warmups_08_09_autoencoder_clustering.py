"""TL;DR: Train autoencoders for reconstruction-error novelty detection and K-means clustering.

Inputs: autoencoder_img.zip (normal200, normal, flip) and Directions01.zip.
Outputs: Reconstruction figures, ROC-AUC/logs, PCA cluster plots, and cluster-assignment CSVs.

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

    # Locate both datasets, seed training, and initialize logging.
    # Setup, Drive mount, archive discovery, and output logging

    import os, random, shutil, time, zipfile
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    SEED = 507
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DRIVE = args.data_dir
    OUT = args.output_dir / "exercise_8_9_output"
    OUT.mkdir(parents=True, exist_ok=True)
    WORK = args.work_dir
    WORK.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUT / "warmup_exercises_8_9_log.txt"
    log_lines = []

    def log(message=""):
        """Print an experiment message and persist it in the current run log."""
        print(message)
        log_lines.append(str(message))
        LOG_PATH.write_text("\n".join(log_lines) + "\n")

    def locate_archive(name):
        """Find an exact archive filename under the configured input directory."""
        matches = [
            p
            for p in DRIVE.rglob("*")
            if p.is_file() and p.name.lower() == name.lower()
        ]
        if not matches:
            raise FileNotFoundError(f"Could not find {name} below {DRIVE}")
        return matches[0]

    def extract_archive(archive, destination):
        """Extract into a dedicated scratch directory when it is empty."""
        if not destination.exists() or not any(destination.iterdir()):
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                z.extractall(destination)

    AE_ZIP = locate_archive("autoencoder_img.zip")
    DIR_ZIP = locate_archive("Directions01.zip")
    extract_archive(AE_ZIP, WORK / "autoencoder_img")
    extract_archive(DIR_ZIP, WORK / "directions")
    AE_ROOT = next(
        p for p in (WORK / "autoencoder_img").rglob("autoencoder_img") if p.is_dir()
    )
    DIR_ROOT = next(
        p for p in (WORK / "directions").rglob("Directions01") if p.is_dir()
    )
    log("Warm-up Exercises 8 and 9 experiment log")
    log(
        f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')} | device: {DEVICE} | seed: {SEED}"
    )
    log(f"Archives: {AE_ZIP.name}, {DIR_ZIP.name}")
    log(f"Output directory: {OUT}")

    # Define 64x64 grayscale preprocessing, 64-dimensional latent features, and MSE training.
    # Shared image and autoencoder utilities
    VALID = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def image_files(folder):
        # A BMP is preferred when equivalent BMP/TIFF copies share the same case ID.
        """Find supported images, preferring BMP over duplicate TIFF encodings."""
        candidates = sorted(
            p for p in Path(folder).rglob("*") if p.suffix.lower() in VALID
        )
        chosen = {}
        priority = {".bmp": 0, ".png": 1, ".jpg": 2, ".jpeg": 2, ".tif": 3, ".tiff": 3}
        for p in candidates:
            key = str(p.with_suffix(""))
            if (
                key not in chosen
                or priority[p.suffix.lower()] < priority[chosen[key].suffix.lower()]
            ):
                chosen[key] = p
        return list(chosen.values())

    def read_image(path, size=64):
        """Resize to grayscale float32 in [0, 1] for autoencoder reconstruction."""
        im = (
            Image.open(path)
            .convert("L")
            .resize((size, size), Image.Resampling.BILINEAR)
        )
        return np.asarray(im, dtype=np.float32) / 255.0

    class ImageSet(Dataset):
        def __init__(self, paths):
            """Initialize the dataset index or model layers using the supplied settings."""
            self.paths = list(paths)

        def __len__(self):
            """Return the number of indexed samples."""
            return len(self.paths)

        def __getitem__(self, i):
            """Load and transform one indexed sample for the training data loader."""
            return torch.from_numpy(read_image(self.paths[i]))[None, ...]

    class ConvAutoencoder(nn.Module):
        def __init__(self, zdim=64):
            """Initialize the dataset index or model layers using the supplied settings."""
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, zdim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(zdim, 64 * 8 * 8),
                nn.ReLU(),
                nn.Unflatten(1, (64, 8, 8)),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            """Apply the model and return its prediction (plus latent features for autoencoders)."""
            z = self.encoder(x)
            return self.decoder(z), z

    def train_autoencoder(paths, epochs=30, batch_size=32, lr=1e-3, name="model"):
        """Fit reconstruction MSE using unlabeled inputs and log mean training loss."""
        model = ConvAutoencoder().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses = []
        loader = DataLoader(
            ImageSet(paths),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(DEVICE == "cuda"),
        )
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0
            for x in loader:
                x = x.to(DEVICE)
                out, _ = model(x)
                loss = loss_fn(out, x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            value = total / len(paths)
            losses.append(value)
            log(f"{name} epoch {epoch:02d}: train reconstruction MSE = {value:.6f}")
        return model, losses

    @torch.no_grad()
    def embed_and_score(model, paths):
        """Evaluate without gradients; return latent vectors, per-image MSE, inputs, and reconstructions."""
        loader = DataLoader(
            ImageSet(paths), batch_size=32, shuffle=False, num_workers=0
        )
        model.eval()
        zs = []
        scores = []
        originals = []
        reconstructions = []
        for x in loader:
            x = x.to(DEVICE)
            out, z = model(x)
            zs.append(z.cpu().numpy())
            scores.extend(((out - x) ** 2).mean((1, 2, 3)).cpu().numpy())
            originals.extend(x.cpu().numpy()[:, 0])
            reconstructions.extend(out.cpu().numpy()[:, 0])
        return (
            np.vstack(zs),
            np.asarray(scores),
            np.asarray(originals),
            np.asarray(reconstructions),
        )

    # Exercise 8: train on normal images only; derive the threshold from training errors.
    # Exercise 8 — anomaly/novelty detection using reconstruction error
    train_normal = image_files(AE_ROOT / "normal200")
    test_normal = image_files(AE_ROOT / "normal")
    anomalies = image_files(AE_ROOT / "flip")
    log("\nEXERCISE 8: ANOMALY / NOVELTY DETECTION")
    log(
        f"Normal training images: {len(train_normal)} | held-out normal: {len(test_normal)} | injected/irregular (flip): {len(anomalies)}"
    )
    e8_model, e8_losses = train_autoencoder(train_normal, epochs=30, name="Exercise 8")
    z_train, score_train, _, _ = embed_and_score(e8_model, train_normal)
    z_normal, score_normal, normal_x, normal_hat = embed_and_score(
        e8_model, test_normal
    )
    z_anom, score_anom, anom_x, anom_hat = embed_and_score(e8_model, anomalies)
    threshold = np.percentile(score_train, 95)
    y = np.r_[np.zeros(len(score_normal)), np.ones(len(score_anom))]
    s = np.r_[score_normal, score_anom]
    auc = roc_auc_score(y, s)
    log(f"95th-percentile normal-training threshold: {threshold:.6f}")
    log(
        f"Held-out normal mean error: {score_normal.mean():.6f}; flip mean error: {score_anom.mean():.6f}; ROC-AUC: {auc:.3f}"
    )
    log(
        f"Flagged as novel: normal {np.mean(score_normal > threshold):.1%}; flip {np.mean(score_anom > threshold):.1%}"
    )
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(1, 31), e8_losses, marker="o")
    ax[0].set(
        title="Exercise 8: autoencoder training",
        xlabel="Epoch",
        ylabel="Reconstruction MSE",
    )
    ax[0].grid(alpha=0.3)
    ax[1].hist(score_normal, bins=12, alpha=0.75, label="held-out normal")
    ax[1].hist(score_anom, bins=12, alpha=0.75, label="flip / injected")
    ax[1].axvline(threshold, color="k", ls="--", label="95% train threshold")
    ax[1].set(
        title=f"Novelty scores (AUC={auc:.3f})",
        xlabel="Per-image reconstruction MSE",
        ylabel="Images",
    )
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "exercise_8_novelty_scores.png", dpi=180)
    plt.show()
    # Representative reconstruction comparison: highest-scoring normal and anomaly.
    ni = np.argmax(score_normal)
    ai = np.argmax(score_anom)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, (x, hat, title, score) in enumerate(
        [
            (normal_x[ni], normal_hat[ni], "Held-out normal", score_normal[ni]),
            (anom_x[ai], anom_hat[ai], "Novel / flipped image", score_anom[ai]),
        ]
    ):
        for a, img, label in zip(
            axes[row],
            [x, hat, np.abs(x - hat)],
            ["input", "reconstruction", "absolute error"],
        ):
            a.imshow(img, cmap="gray")
            a.set_title(label)
            a.axis("off")
        axes[row, 0].set_ylabel(f"{title}\nMSE={score:.5f}")
    fig.suptitle("Exercise 8: reconstruction-error novelty detection")
    fig.tight_layout()
    fig.savefig(OUT / "exercise_8_examples.png", dpi=180)
    plt.show()

    # Exercise 9: cluster standardized embeddings without labels; use labels only for ARI/NMI auditing. PCA is visualization only.
    # Exercise 9 — unsupervised clustering (Directions01 and autoencoder image embeddings)
    def cluster_report(features, labels, name, n_clusters, paths):
        """Fit K-means without reference labels, then audit ARI/NMI and export a PCA view plus assignments."""
        x = StandardScaler().fit_transform(features)
        km = KMeans(n_clusters=n_clusters, n_init=30, random_state=SEED).fit(x)
        clusters = km.labels_
        counts = pd.Series(clusters).value_counts().sort_index()
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)
        log(
            f"{name}: K={n_clusters}; cluster sizes={counts.to_dict()}; balanced ratio min/max={counts.min() / counts.max():.3f}; ARI={ari:.3f}; NMI={nmi:.3f}"
        )
        pca = PCA(n_components=2, random_state=SEED).fit_transform(x)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(pca[:, 0], pca[:, 1], c=clusters, cmap="tab10", s=28, alpha=0.8)
        for c in range(n_clusters):
            idx = np.where(clusters == c)[0]
            ax.scatter(
                pca[idx, 0].mean(), pca[idx, 1].mean(), marker="x", s=120, c="black"
            )
        ax.set(
            title=f"{name}: unsupervised K-means clusters",
            xlabel="PCA component 1",
            ylabel="PCA component 2",
        )
        fig.colorbar(sc, label="cluster")
        fig.tight_layout()
        filename = (
            "exercise_9_"
            + name.lower().replace(" ", "_").replace("/", "_")
            + "_clusters.png"
        )
        fig.savefig(OUT / filename, dpi=180)
        plt.show()
        pd.DataFrame(
            {
                "path": [str(p) for p in paths],
                "reference_label": labels,
                "cluster": clusters,
            }
        ).to_csv(OUT / (filename.replace(".png", ".csv")), index=False)
        return clusters

    log("\nEXERCISE 9: UNSUPERVISED CLUSTERING")
    # Directions: train an autoencoder on all training images without their up/down/left/right labels.
    direction_train = []
    direction_test = []
    for split, collection in [("train", direction_train), ("test", direction_test)]:
        for label_dir in sorted((DIR_ROOT / split).iterdir()):
            if label_dir.is_dir():
                collection.extend((p, label_dir.name) for p in image_files(label_dir))
    d_train_paths = [p for p, _ in direction_train]
    d_test_paths = [p for p, _ in direction_test]
    d_test_labels = [y for _, y in direction_test]
    log(
        f"Directions01: unlabeled clustering train/test images = {len(d_train_paths)}/{len(d_test_paths)}; audit classes = {sorted(set(d_test_labels))}"
    )
    direction_model, _ = train_autoencoder(
        d_train_paths, epochs=25, name="Exercise 9 Directions"
    )
    d_z, _, _, _ = embed_and_score(direction_model, d_test_paths)
    direction_clusters = cluster_report(
        d_z, d_test_labels, "Directions01", 4, d_test_paths
    )

    # The same analysis on autoencoder_img: normal versus flipped is withheld until the audit step.
    ae_paths = test_normal + anomalies
    ae_labels = ["normal"] * len(test_normal) + ["flip"] * len(anomalies)
    ae_z = np.vstack([z_normal, z_anom])
    ae_clusters = cluster_report(ae_z, ae_labels, "Autoencoder image set", 2, ae_paths)

    # Direction cluster mosaic: lets the author visually inspect what each group captured.
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    rng = np.random.default_rng(SEED)
    for c in range(4):
        idx = np.where(direction_clusters == c)[0]
        chosen = rng.choice(idx, size=min(4, len(idx)), replace=False)
        for j, a in enumerate(axes[c]):
            if j < len(chosen):
                a.imshow(read_image(d_test_paths[chosen[j]]), cmap="gray")
                a.set_title(f"cluster {c}")
            a.axis("off")
    fig.suptitle("Exercise 9: representative Directions01 cluster members")
    fig.tight_layout()
    fig.savefig(OUT / "exercise_9_directions_mosaic.png", dpi=180)
    plt.show()
    log(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Saved log: {LOG_PATH}")
    log("Saved figures and CSV cluster assignments in: " + str(OUT))


if __name__ == "__main__":
    main()
