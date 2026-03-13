"""
rag/cxr_loader.py — CXR Dataset Loader & Reference Guide
=========================================================
Provides download instructions, directory setup, and a simple loader
for three public CXR datasets used for testing MedAI:

  1. NIH ChestX-ray14   — 112,120 PNG images, 14 pathology labels
  2. CheXpert            — 224,316 images with uncertainty labels + reports
  3. VinDr-CXR          — 18,000 images with radiologist bounding boxes

Usage:
    python rag/cxr_loader.py --dataset nih --limit 20
    python rag/cxr_loader.py --dataset chexpert --limit 20
    python rag/cxr_loader.py --list
"""

import os
import json
import argparse
import sys

# ── Dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    "nih": {
        "name": "NIH ChestX-ray14",
        "size": "112,120 PNG images",
        "labels": 14,
        "pathologies": [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ],
        "format": "PNG",
        "access": "Public — NIH Clinical Center",
        "download_url": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "gcp_mirror": "gs://gcs-public-data--healthcare-nih-chest-xray/png/",
        "paper": "https://arxiv.org/abs/1705.02315",
        "local_dir": "sample_data/images/nih/",
        "instructions": [
            "1. Visit https://nihcc.app.box.com/v/ChestXray-NIHCC",
            "2. Download images_001.tar.gz through images_012.tar.gz (42GB total)",
            "   OR use GCP: gsutil -m cp 'gs://gcs-public-data--healthcare-nih-chest-xray/png/*.png' sample_data/images/nih/",
            "3. Extract to sample_data/images/nih/",
            "4. Download Data_Entry_2017.csv for labels"
        ],
        "mini_subset": "For quick testing, download only images_001.tar.gz (~3.5GB, ~9000 images)"
    },
    "chexpert": {
        "name": "CheXpert",
        "size": "224,316 images with radiology reports",
        "labels": 14,
        "pathologies": [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"
        ],
        "format": "JPEG",
        "access": "Free registration required — Stanford ML Group",
        "download_url": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "paper": "https://arxiv.org/abs/1901.07031",
        "local_dir": "sample_data/images/chexpert/",
        "instructions": [
            "1. Register at https://stanfordmlgroup.github.io/competitions/chexpert/",
            "2. Download CheXpert-v1.0-small.zip (11GB) for downsampled version",
            "   OR CheXpert-v1.0.zip (439GB) for full resolution",
            "3. Extract to sample_data/images/chexpert/",
            "4. train.csv and valid.csv contain labels and uncertainty markers"
        ],
        "mini_subset": "CheXpert-v1.0-small contains 224x224 images suitable for quick testing"
    },
    "vindr": {
        "name": "VinDr-CXR",
        "size": "18,000 DICOM images with radiologist bounding boxes",
        "labels": 28,
        "pathologies": [
            "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
            "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
            "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung tumor",
            "Mediastinal shift", "Nodule/Mass", "Other lesion", "Pleural effusion",
            "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
            "Rib fracture", "COPD", "Lung lesion", "Lung tumor", "No finding"
        ],
        "format": "DICOM",
        "access": "Free — PhysioNet (requires credentialed account)",
        "download_url": "https://physionet.org/content/vindr-cxr/1.0.0/",
        "paper": "https://www.nature.com/articles/s41597-022-01498-w",
        "local_dir": "sample_data/images/vindr/",
        "instructions": [
            "1. Create a PhysioNet account at https://physionet.org/register/",
            "2. Complete required training and request access to VinDr-CXR",
            "3. Download via: wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/vindr-cxr/1.0.0/",
            "4. Extract DICOM files to sample_data/images/vindr/",
            "5. annotations/annotations_train.csv contains bounding box coordinates"
        ],
        "mini_subset": "VinDr provides a 3,000-image test set (~15GB) which is ideal for overlay demonstrations",
        "overlay_note": "Bounding box format: x_min, y_min, x_max, y_max in pixel coordinates — convert to fractional for MedAI overlays"
    }
}


def list_datasets():
    """Print all available datasets with download info."""
    print("\n" + "="*60)
    print("MedAI — CXR Dataset Reference Guide")
    print("="*60)
    for key, ds in DATASETS.items():
        print(f"\n[{key.upper()}] {ds['name']}")
        print(f"  Size:    {ds['size']}")
        print(f"  Format:  {ds['format']}")
        print(f"  Access:  {ds['access']}")
        print(f"  URL:     {ds['download_url']}")
        print(f"  Paper:   {ds['paper']}")
        print(f"  Tip:     {ds.get('mini_subset', '')}")
    print("\n" + "="*60)


def setup_dirs():
    """Create local image directories."""
    for ds in DATASETS.values():
        os.makedirs(ds["local_dir"], exist_ok=True)
    print("✅ Created image directories:")
    for ds in DATASETS.values():
        print(f"   {ds['local_dir']}")


def print_instructions(dataset_key: str):
    """Print step-by-step download instructions for a dataset."""
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}. Choose from: {list(DATASETS.keys())}")
        sys.exit(1)
    ds = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Download instructions: {ds['name']}")
    print(f"{'='*60}")
    for step in ds["instructions"]:
        print(f"  {step}")
    if "overlay_note" in ds:
        print(f"\n  Note: {ds['overlay_note']}")
    print()


def load_images(dataset_key: str, limit: int = 10) -> list[dict]:
    """
    Load image paths from a local dataset directory.
    Returns list of dicts with path, filename, and dataset metadata.
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return []

    ds = DATASETS[dataset_key]
    local_dir = ds["local_dir"]

    if not os.path.exists(local_dir):
        print(f"Directory not found: {local_dir}")
        print(f"Run: python rag/cxr_loader.py --setup")
        print(f"Then follow download instructions: python rag/cxr_loader.py --instructions {dataset_key}")
        return []

    extensions = {
        "nih": [".png", ".PNG"],
        "chexpert": [".jpg", ".jpeg", ".JPG"],
        "vindr": [".dcm", ".DCM"]
    }.get(dataset_key, [".png", ".jpg", ".dcm"])

    images = []
    for fname in sorted(os.listdir(local_dir)):
        if any(fname.endswith(ext) for ext in extensions):
            images.append({
                "path": os.path.join(local_dir, fname),
                "filename": fname,
                "dataset": ds["name"],
                "format": ds["format"],
                "modality": "CXR"
            })
        if len(images) >= limit:
            break

    print(f"✅ Loaded {len(images)} images from {ds['name']} ({local_dir})")
    return images


def load_for_medai(dataset_key: str, limit: int = 5) -> list[dict]:
    """
    Load images in MedAI-compatible format for batch testing.
    Returns list of dicts ready to pass to the pipeline.
    """
    from PIL import Image as PILImage

    raw = load_images(dataset_key, limit=limit)
    results = []

    for item in raw:
        try:
            if item["format"] == "DICOM":
                import pydicom
                dcm = pydicom.dcmread(item["path"])
                import numpy as np
                arr = dcm.pixel_array.astype(float)
                arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype("uint8")
                pil_img = PILImage.fromarray(arr).convert("RGB")
            else:
                pil_img = PILImage.open(item["path"]).convert("RGB")

            results.append({
                "filename": item["filename"],
                "dataset": item["dataset"],
                "image_pil": pil_img,
                "size": pil_img.size
            })
            print(f"  Loaded: {item['filename']} {pil_img.size}")
        except Exception as e:
            print(f"  Skipped {item['filename']}: {e}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAI CXR Dataset Loader")
    parser.add_argument("--list",         action="store_true",  help="List all datasets")
    parser.add_argument("--setup",        action="store_true",  help="Create local image directories")
    parser.add_argument("--instructions", metavar="DATASET",    help="Show download instructions (nih/chexpert/vindr)")
    parser.add_argument("--dataset",      metavar="DATASET",    help="Load images from dataset (nih/chexpert/vindr)")
    parser.add_argument("--limit",        type=int, default=10, help="Max images to load (default: 10)")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.setup:
        setup_dirs()
    elif args.instructions:
        print_instructions(args.instructions)
    elif args.dataset:
        imgs = load_images(args.dataset, limit=args.limit)
        if imgs:
            print(f"\nFirst {min(3, len(imgs))} entries:")
            for img in imgs[:3]:
                print(f"  {img}")
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python rag/cxr_loader.py --list")
        print("  python rag/cxr_loader.py --setup")
        print("  python rag/cxr_loader.py --instructions nih")
        print("  python rag/cxr_loader.py --dataset nih --limit 20")