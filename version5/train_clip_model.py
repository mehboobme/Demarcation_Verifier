#!/usr/bin/env python3
"""
Train CLIP Model on ROSHN Reference Images
==========================================
Trains CLIP classifier on facade types and floor plans with header text analysis.

Usage:
    python train_clip_model.py
    python train_clip_model.py --reference-dir ./reference_images --output ./models/roshn_classifier.pkl
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.error("CLIP not installed. Install with:")
    logger.error("  pip install torch torchvision")
    logger.error("  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

try:
    import pickle
except ImportError:
    logger.error("pickle not available")
    sys.exit(1)


class ROSHNClassifier:
    """
    CLIP-based classifier for ROSHN demarcation plans.
    
    Features:
    - Facade types: traditional vs modern
    - Floor plan types: traditional vs modern layouts
    """
    
    def __init__(self, model_name: str = "ViT-B/32", use_gpu: bool = False):
        """
        Initialize CLIP classifier.
        
        Args:
            model_name: CLIP model variant
                - "ViT-B/32" - Fast, good accuracy (default)
                - "ViT-B/16" - Slower, better accuracy
                - "ViT-L/14" - Best accuracy, slower
            use_gpu: Enable GPU acceleration if available (RTX 5070)
                NOTE: Disabled by default due to RTX 5070 requiring CUDA 12.8+
        """
        # Force CPU for now - RTX 5070 (sm_120) requires PyTorch with CUDA 12.8/13.0
        self.device = "cpu"
        logger.info(f"Using device: {self.device} (RTX 5070 requires PyTorch CUDA 12.8+, using CPU)")
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_name = model_name
        
        # Storage for embeddings
        self.embeddings = {}  # {feature_type: {category: np.array}}
        self.image_paths = {}  # {feature_type: {category: [paths]}}
        self.image_metadata = {}  # {feature_type: {category: [metadata_dicts]}}
    
    def _encode_image(self, image_path: str) -> np.ndarray:
        """Convert image to CLIP embedding."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to encode {image_path}: {e}")
            return None
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract metadata from filename.
        Example: "traditional_front_elevation_villa_type_a.jpg"
        """
        metadata = {
            'filename': filename,
            'has_traditional': 'traditional' in filename.lower(),
            'has_modern': 'modern' in filename.lower(),
            'has_front': 'front' in filename.lower(),
            'has_rear': 'rear' in filename.lower(),
            'has_side': 'side' in filename.lower(),
            'has_elevation': 'elevation' in filename.lower(),
            'has_plan': 'plan' in filename.lower() or 'floor' in filename.lower(),
        }
        return metadata
    
    def train(self, reference_dir: str):
        """
        Train classifier on reference images.
        
        Expected structure:
            reference_dir/
                facade_types/
                    traditional/
                        img1.jpg, img2.jpg, ...
                    modern/
                        img1.jpg, ...
                floor_plans/
                    traditional/
                        plan1.jpg, ...
                    modern/
                        plan1.jpg, ...
        
        Args:
            reference_dir: Path to organized reference images
        """
        logger.info("="*70)
        logger.info("TRAINING ROSHN CLIP CLASSIFIER")
        logger.info("="*70)
        
        ref_path = Path(reference_dir)
        
        if not ref_path.exists():
            logger.error(f"Reference directory not found: {reference_dir}")
            return False
        
        # Define expected structure
        feature_mapping = {
            'facade_types': 'facade_type',
            'floor_plans': 'floor_plan_type'
        }
        
        total_images = 0
        
        for dir_name, feature_type in feature_mapping.items():
            feature_dir = ref_path / dir_name
            
            if not feature_dir.exists():
                logger.warning(f"Directory not found: {feature_dir}")
                continue
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {feature_type}")
            logger.info(f"{'='*70}")
            
            self.embeddings[feature_type] = {}
            self.image_paths[feature_type] = {}
            self.image_metadata[feature_type] = {}
            
            # Process each category (traditional/modern)
            for category_dir in feature_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                category = category_dir.name
                logger.info(f"\n  📁 Category: {category}")
                
                # Find all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(category_dir.glob(ext)))
                
                if not image_files:
                    logger.warning(f"    ⚠️  No images found in {category_dir}")
                    continue
                
                # Process each image
                embeddings = []
                paths = []
                metadata_list = []
                
                logger.info(f"    Processing {len(image_files)} images...")
                
                for img_path in sorted(image_files):
                    # Extract metadata from filename
                    metadata = self._extract_metadata_from_filename(img_path.name)
                    
                    # Encode image
                    emb = self._encode_image(str(img_path))
                    
                    if emb is not None:
                        embeddings.append(emb)
                        paths.append(str(img_path))
                        metadata_list.append(metadata)
                        logger.info(f"      ✓ {img_path.name}")
                    else:
                        logger.warning(f"      ✗ Failed: {img_path.name}")
                
                # Store embeddings
                if embeddings:
                    self.embeddings[feature_type][category] = np.array(embeddings)
                    self.image_paths[feature_type][category] = paths
                    self.image_metadata[feature_type][category] = metadata_list
                    
                    total_images += len(embeddings)
                    logger.info(f"    ✓ Loaded {len(embeddings)} images for '{category}'")
                else:
                    logger.warning(f"    ⚠️  No valid embeddings for '{category}'")
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        self._print_summary()
        logger.info(f"\n📊 Total images processed: {total_images}")
        logger.info("="*70)
        
        return total_images > 0
    
    def _print_summary(self):
        """Print training summary."""
        for feature_type, categories in self.embeddings.items():
            if categories:
                logger.info(f"\n{feature_type}:")
                for category, embs in categories.items():
                    count = len(embs)
                    logger.info(f"  • {category}: {count} images")
    
    def classify(
        self,
        image_path: str,
        top_k: int = 5,
        features: List[str] = None,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Classify an image.
        
        Args:
            image_path: Path to image
            top_k: Number of nearest neighbors
            features: Features to classify (default: all)
            confidence_threshold: Minimum confidence to accept result
        
        Returns:
            {
                'facade_type': {
                    'category': 'traditional',
                    'confidence': 0.92,
                    'matches': [...]
                },
                'floor_plan_type': {...}
            }
        """
        if features is None:
            features = list(self.embeddings.keys())
        
        # Encode query image
        query_emb = self._encode_image(image_path)
        if query_emb is None:
            return {}
        
        results = {}
        
        for feature_type in features:
            if feature_type not in self.embeddings:
                continue
            
            if not self.embeddings[feature_type]:
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
            
            # Majority vote with weighted confidence
            votes = {}
            conf_sum = {}
            
            for match in top_matches:
                cat = match['category']
                sim = match['similarity']
                votes[cat] = votes.get(cat, 0) + 1
                conf_sum[cat] = conf_sum.get(cat, 0.0) + sim
            
            # Winner
            predicted = max(votes, key=votes.get)
            
            # Confidence: average similarity of winning category
            winning = [m for m in top_matches if m['category'] == predicted]
            confidence = sum(m['similarity'] for m in winning) / len(winning)
            
            results[feature_type] = {
                'category': predicted,
                'confidence': float(confidence),
                'votes': votes,
                'matches': top_matches
            }
        
        return results
    
    def save(self, filepath: str):
        """Save trained model."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        data = {
            'model_name': self.model_name,
            'embeddings': self.embeddings,
            'image_paths': self.image_paths,
            'image_metadata': self.image_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"\n✓ Model saved: {filepath}")
        logger.info(f"  Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    
    @classmethod
    def load(cls, filepath: str) -> 'ROSHNClassifier':
        """Load trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(model_name=data['model_name'])
        classifier.embeddings = data['embeddings']
        classifier.image_paths = data['image_paths']
        classifier.image_metadata = data.get('image_metadata', {})
        
        logger.info(f"✓ Model loaded: {filepath}")
        classifier._print_summary()
        
        return classifier
    
    def is_trained(self) -> bool:
        """Check if classifier has been trained."""
        return bool(self.embeddings)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train CLIP classifier on ROSHN reference images'
    )
    parser.add_argument(
        '--reference-dir',
        type=str,
        default='./reference_images',
        help='Directory containing reference images (default: ./reference_images)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./models/roshn_classifier.pkl',
        help='Output path for trained model (default: ./models/roshn_classifier.pkl)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50'],
        help='CLIP model variant (default: ViT-B/32)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ROSHN CLIP CLASSIFIER TRAINING")
    print("="*70)
    print(f"Reference directory: {args.reference_dir}")
    print(f"Output model: {args.output}")
    print(f"CLIP model: {args.model}")
    print("="*70)
    
    # Check reference directory
    ref_path = Path(args.reference_dir)
    if not ref_path.exists():
        logger.error(f"Reference directory not found: {args.reference_dir}")
        logger.info("\nExpected structure:")
        logger.info(f"  {args.reference_dir}/")
        logger.info("    facade_types/")
        logger.info("      traditional/")
        logger.info("        img1.jpg, img2.jpg, ...")
        logger.info("      modern/")
        logger.info("        img1.jpg, ...")
        logger.info("    floor_plans/")
        logger.info("      traditional/")
        logger.info("        plan1.jpg, ...")
        logger.info("      modern/")
        logger.info("        plan1.jpg, ...")
        sys.exit(1)
    
    # Check for images
    has_images = False
    for feature_dir in ['facade_types', 'floor_plans']:
        feature_path = ref_path / feature_dir
        if feature_path.exists():
            for category_dir in feature_path.iterdir():
                if category_dir.is_dir():
                    images = (
                        list(category_dir.glob('*.jpg')) +
                        list(category_dir.glob('*.jpeg')) +
                        list(category_dir.glob('*.png'))
                    )
                    if images:
                        has_images = True
                        break
    
    if not has_images:
        logger.error(f"No images found in {args.reference_dir}")
        logger.info("Please add your reference images and try again.")
        sys.exit(1)
    
    # Train classifier
    try:
        classifier = ROSHNClassifier(model_name=args.model)
        success = classifier.train(args.reference_dir)
        
        if not success:
            logger.error("Training failed - no images processed")
            sys.exit(1)
        
        classifier.save(args.output)
        
        print("\n" + "="*70)
        print("✓ SUCCESS!")
        print("="*70)
        print(f"\nModel saved to: {args.output}")
        print("\nYou can now use this model for automatic classification:")
        print("  1. In your code:")
        print(f"     classifier = ROSHNClassifier.load('{args.output}')")
        print("     result = classifier.classify('image.jpg')")
        print("\n  2. Test the model:")
        print(f"     python test_classifier.py {args.output} <test_image.jpg>")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
