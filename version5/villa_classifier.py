"""
Automatic Villa Feature Classifier using CLIP
=============================================
Classifies villa features (facade type, render type, view type) using reference images.
No API calls needed after initial training.

Author: Claude Code
Date: 2025-12-01
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Check if CLIP is available
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")


class VillaClassifier:
    """
    Automatic villa feature classifier using CLIP embeddings.

    Features:
    - Facade type: traditional vs modern
    - Render type: white_render, stone, brick, painted
    - View type: front, rear, side

    Usage:
        # One-time training
        classifier = VillaClassifier()
        classifier.train('./reference_images')
        classifier.save('./models/villa_classifier.pkl')

        # Use for classification
        classifier = VillaClassifier.load('./models/villa_classifier.pkl')
        result = classifier.classify('./villa.jpg')
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize classifier.

        Args:
            model_name: CLIP model variant
                - "ViT-B/32" - Fast, good accuracy (default)
                - "ViT-B/16" - Slower, better accuracy
                - "RN50" - ResNet-50 based
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP not installed. Install with:\n"
                "pip install torch torchvision\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_name = model_name

        # Reference embeddings storage
        self.embeddings = {}  # {feature_type: {category: np.array}}
        self.image_paths = {}  # {feature_type: {category: [paths]}}

    def _encode_image(self, image_path: str) -> np.ndarray:
        """Convert image to CLIP embedding."""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()

    def train(self, reference_dir: str):
        """
        Create embeddings from reference images.

        Directory structure:
            reference_dir/
                facade_type/
                    traditional/
                        img1.jpg, img2.jpg, ...
                    modern/
                        img1.jpg, ...
                render_type/
                    white_render/
                        img1.jpg, ...
                    stone/
                        img1.jpg, ...
                view_type/
                    front/
                        img1.jpg, ...
                    rear/
                        img1.jpg, ...
                    side/
                        img1.jpg, ...

        Args:
            reference_dir: Path to organized reference images
        """
        logger.info("="*60)
        logger.info("TRAINING VILLA CLASSIFIER")
        logger.info("="*60)

        ref_path = Path(reference_dir)

        for feature_type in ['facade_type', 'render_type', 'view_type']:
            feature_dir = ref_path / feature_type

            if not feature_dir.exists():
                logger.warning(f"Directory not found: {feature_dir}")
                continue

            logger.info(f"\nProcessing {feature_type}...")

            self.embeddings[feature_type] = {}
            self.image_paths[feature_type] = {}

            for category_dir in feature_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category = category_dir.name
                logger.info(f"  Category: {category}")

                # Find all images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(category_dir.glob(ext)))

                if not image_files:
                    logger.warning(f"    No images found in {category_dir}")
                    continue

                # Create embeddings
                embeddings = []
                paths = []

                for img_path in image_files:
                    try:
                        emb = self._encode_image(str(img_path))
                        embeddings.append(emb)
                        paths.append(str(img_path))
                    except Exception as e:
                        logger.error(f"    Failed to process {img_path.name}: {e}")

                if embeddings:
                    self.embeddings[feature_type][category] = np.array(embeddings)
                    self.image_paths[feature_type][category] = paths
                    logger.info(f"    Loaded {len(embeddings)} images")

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        self._print_summary()
        logger.info("="*60)

    def _print_summary(self):
        """Print training summary."""
        total = 0
        for feature_type, categories in self.embeddings.items():
            if categories:
                logger.info(f"\n{feature_type}:")
                for category, embs in categories.items():
                    count = len(embs)
                    total += count
                    logger.info(f"  {category}: {count} images")
        logger.info(f"\nTotal: {total} reference images")

    def classify(
        self,
        image_path: str,
        top_k: int = 5,
        features: Optional[List[str]] = None
    ) -> Dict:
        """
        Classify a villa image.

        Args:
            image_path: Path to villa image
            top_k: Number of nearest neighbors to consider
            features: Features to classify (default: all)

        Returns:
            {
                'facade_type': {
                    'category': 'traditional',
                    'confidence': 0.92,
                    'matches': [
                        {'category': 'traditional', 'similarity': 0.95, 'image': '...'},
                        ...
                    ]
                },
                'render_type': {...},
                'view_type': {...}
            }
        """
        if features is None:
            features = list(self.embeddings.keys())

        # Encode query image
        query_emb = self._encode_image(image_path)

        results = {}

        for feature_type in features:
            if feature_type not in self.embeddings:
                logger.warning(f"No reference data for {feature_type}")
                continue

            if not self.embeddings[feature_type]:
                logger.warning(f"No categories in {feature_type}")
                continue

            # Find nearest neighbors
            all_matches = []

            for category, ref_embs in self.embeddings[feature_type].items():
                # Cosine similarity
                similarities = ref_embs @ query_emb

                category_paths = self.image_paths[feature_type][category]

                for sim, path in zip(similarities, category_paths):
                    all_matches.append({
                        'category': category,
                        'similarity': float(sim),
                        'image': path
                    })

            # Sort by similarity
            all_matches.sort(key=lambda x: x['similarity'], reverse=True)
            top_matches = all_matches[:top_k]

            # Majority vote
            votes = {}
            conf_sum = {}

            for match in top_matches:
                cat = match['category']
                sim = match['similarity']
                votes[cat] = votes.get(cat, 0) + 1
                conf_sum[cat] = conf_sum.get(cat, 0.0) + sim

            # Winner
            predicted = max(votes, key=votes.get)

            # Confidence: avg similarity of winning category
            winning = [m for m in top_matches if m['category'] == predicted]
            confidence = sum(m['similarity'] for m in winning) / len(winning)

            results[feature_type] = {
                'category': predicted,
                'confidence': float(confidence),
                'matches': top_matches
            }

        return results

    def save(self, filepath: str):
        """Save trained model."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        data = {
            'model_name': self.model_name,
            'embeddings': self.embeddings,
            'image_paths': self.image_paths
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'VillaClassifier':
        """Load trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        classifier = cls(model_name=data['model_name'])
        classifier.embeddings = data['embeddings']
        classifier.image_paths = data['image_paths']

        logger.info(f"Model loaded: {filepath}")
        classifier._print_summary()

        return classifier

    def is_trained(self) -> bool:
        """Check if classifier has been trained."""
        return bool(self.embeddings)


# Utility functions

def setup_reference_structure(base_dir: str = "./reference_images"):
    """Create directory structure for reference images."""
    structure = {
        'facade_type': ['traditional', 'modern'],
        'render_type': ['white_render', 'stone', 'brick', 'painted'],
        'view_type': ['front', 'rear', 'side']
    }

    base_path = Path(base_dir)

    for feature_type, categories in structure.items():
        for category in categories:
            dir_path = base_path / feature_type / category
            dir_path.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Directory structure created: {base_dir}")
    print("\nPlace your reference images in:")
    for feature_type, categories in structure.items():
        print(f"\n{feature_type}:")
        for category in categories:
            print(f"  {base_dir}/{feature_type}/{category}/")

    return base_path
