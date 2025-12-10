#!/usr/bin/env python3
"""
Train Villa Classifier
=====================
Train the CLIP-based classifier on your reference images.

Usage:
    python train_classifier.py [reference_dir] [output_model]

Examples:
    python train_classifier.py
    python train_classifier.py ./my_references ./models/classifier.pkl
"""

import sys
import logging
from pathlib import Path
from villa_classifier import VillaClassifier, setup_reference_structure

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def main():
    print("="*60)
    print("VILLA CLASSIFIER TRAINING")
    print("="*60)

    # Parse arguments
    reference_dir = sys.argv[1] if len(sys.argv) > 1 else "./reference_images"
    output_model = sys.argv[2] if len(sys.argv) > 2 else "./models/villa_classifier.pkl"

    # Check if reference directory exists
    ref_path = Path(reference_dir)

    if not ref_path.exists():
        print(f"\n[ERROR] Reference directory not found: {reference_dir}")
        print("\nWould you like to create the directory structure? (y/n)")

        response = input().strip().lower()
        if response == 'y':
            setup_reference_structure(reference_dir)
            print("\n[INFO] Directory structure created!")
            print("[INFO] Please add your reference images and run this script again.")
            return
        else:
            print("[INFO] Training cancelled.")
            return

    # Check for images
    has_images = False
    for feature_dir in ['facade_type', 'render_type', 'view_type']:
        feature_path = ref_path / feature_dir
        if feature_path.exists():
            for category_dir in feature_path.iterdir():
                if category_dir.is_dir():
                    images = list(category_dir.glob('*.jpg')) + \
                            list(category_dir.glob('*.png')) + \
                            list(category_dir.glob('*.jpeg'))
                    if images:
                        has_images = True
                        break

    if not has_images:
        print(f"\n[ERROR] No images found in {reference_dir}")
        print("\nExpected structure:")
        print(f"  {reference_dir}/")
        print("    facade_type/")
        print("      traditional/")
        print("        img1.jpg, img2.jpg, ...")
        print("      modern/")
        print("        img1.jpg, ...")
        print("    render_type/")
        print("      white_render/")
        print("        img1.jpg, ...")
        print("    view_type/")
        print("      front/")
        print("        img1.jpg, ...")
        print("\nPlease add your reference images and try again.")
        return

    # Train classifier
    print(f"\n[INFO] Training on images from: {reference_dir}")
    print(f"[INFO] Will save model to: {output_model}")
    print()

    try:
        classifier = VillaClassifier()
        classifier.train(reference_dir)
        classifier.save(output_model)

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\n[OK] Model trained and saved to: {output_model}")
        print("\nYou can now use this model for automatic classification:")
        print("  1. In GUI: Enable 'Use Automatic Classifier' option")
        print("  2. In code: classifier = VillaClassifier.load('{}')'".format(output_model))

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
