"""
Facade Classification using Vision Model Embeddings + Traditional ML Classifiers
Combines CLIP/DINO/ViT feature extraction with SVM/RandomForest/XGBoost

Advantages:
- Uses powerful pre-trained vision models as feature extractors
- Traditional classifiers work well with small datasets (72 images)
- Much faster training than fine-tuning deep models
- Can use GPU for embedding extraction, CPU for classification
"""

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import CLIP (optional - only needed for CLIP embeddings)
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    clip = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingClassifier:
    """
    Facade classifier using vision model embeddings + traditional ML.
    Supports CLIP, DINO, ViT as feature extractors.
    """
    
    def __init__(self, 
                 embedding_model: str = 'clip',  # 'clip', 'dino', 'vit'
                 classifier_type: str = 'svm',    # 'svm', 'rf', 'xgboost'
                 device: str = 'auto',
                 use_fp16: bool = False,
                 batch_size: int = 1):
        """
        Initialize embedding classifier.
        
        Args:
            embedding_model: Vision model for feature extraction
            classifier_type: Traditional ML classifier
            device: 'cuda', 'cpu', or 'auto'
            use_fp16: Use FP16 mixed precision for faster inference (GPU only)
            batch_size: Batch size for inference (>1 for faster processing)
        """
        self.embedding_model_name = embedding_model
        self.classifier_type = classifier_type
        
        # Auto-detect device - Force GPU if available
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Force GPU for faster processing
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        
        # Optimization settings
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        self.batch_size = batch_size
            
        logger.info(f"Initializing {embedding_model.upper()} + {classifier_type.upper()}")
        logger.info(f"Device: {self.device}")
        if self.use_fp16:
            logger.info("FP16 mixed precision: ENABLED")
        if self.batch_size > 1:
            logger.info(f"Batch size: {self.batch_size}")
        
        # Load embedding model
        self.embedding_model = None
        self.preprocess = None
        self.embedding_dim = None
        self._load_embedding_model()
        
        # Convert to FP16 if enabled
        if self.use_fp16:
            self.embedding_model = self.embedding_model.half()
        
        # Initialize classifier
        self.classifier = None
        self.scaler = StandardScaler()
        self.class_names = ['modern', 'traditional']
        
    def _load_embedding_model(self):
        """Load the vision model for embeddings."""
        if self.embedding_model_name == 'clip':
            self._load_clip()
        elif self.embedding_model_name == 'dino':
            self._load_dino()
        elif self.embedding_model_name == 'vit':
            self._load_vit()
        else:
            raise ValueError(f"Unknown model: {self.embedding_model_name}")
    
    def _load_clip(self):
        """Load CLIP ViT-B/32 model."""
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        logger.info("Loading CLIP ViT-B/32...")
        self.embedding_model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.embedding_dim = 512  # CLIP ViT-B/32 output dimension
        logger.info(f"✓ CLIP loaded (embedding dim: {self.embedding_dim})")
        
    def _load_dino(self):
        """Load DINO ViT model from torch hub."""
        logger.info("Loading DINO ViT-S/16...")
        try:
            self.embedding_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            self.embedding_model = self.embedding_model.to(self.device)
            self.embedding_model.eval()
            self.embedding_dim = 384  # DINO ViT-S/16 output
            
            # DINO preprocessing
            from torchvision import transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"✓ DINO loaded (embedding dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load DINO: {e}")
            # Don't fallback to CLIP if it's not available
            if CLIP_AVAILABLE:
                logger.info("Falling back to CLIP...")
                self._load_clip()
                self.embedding_model_name = 'clip'
            else:
                raise ImportError(f"DINO failed to load and CLIP not available: {e}")
    
    def _load_vit(self):
        """Load Vision Transformer from torchvision."""
        logger.info("Loading ViT-B/16...")
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.embedding_model = vit_b_16(weights=weights)
            self.embedding_model = self.embedding_model.to(self.device)
            self.embedding_model.eval()
            
            # Remove classification head to get embeddings
            self.embedding_model.heads = torch.nn.Identity()
            self.embedding_dim = 768  # ViT-B/16 hidden dimension
            self.preprocess = weights.transforms()
            
            logger.info(f"✓ ViT loaded (embedding dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load ViT: {e}")
            # Don't fallback to CLIP if it's not available
            if CLIP_AVAILABLE:
                logger.info("Falling back to CLIP...")
                self._load_clip()
                self.embedding_model_name = 'clip'
            else:
                raise ImportError(f"ViT failed to load and CLIP not available: {e}")
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding vector from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector (numpy array)
        """
        image = Image.open(image_path).convert('RGB')
        
        if self.embedding_model_name == 'clip':
            # CLIP embedding
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            if self.use_fp16:
                image_input = image_input.half()
            
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        embedding = self.embedding_model.encode_image(image_input)
                else:
                    embedding = self.embedding_model.encode_image(image_input)
                embedding = embedding.cpu().float().numpy().flatten()
                # Normalize (CLIP embeddings are already normalized, but ensure it)
                embedding = embedding / np.linalg.norm(embedding)
                
        elif self.embedding_model_name in ['dino', 'vit']:
            # DINO/ViT embedding
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            if self.use_fp16:
                image_input = image_input.half()
            
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        embedding = self.embedding_model(image_input)
                else:
                    embedding = self.embedding_model(image_input)
                embedding = embedding.cpu().float().numpy().flatten()
                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def extract_embeddings_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract embeddings from multiple images in batches (faster).
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Embedding matrix (N, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_imgs = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                image = Image.open(img_path).convert('RGB')
                img_tensor = self.preprocess(image)
                batch_imgs.append(img_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            if self.use_fp16:
                batch_tensor = batch_tensor.half()
            
            # Extract embeddings
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        if self.embedding_model_name == 'clip':
                            batch_emb = self.embedding_model.encode_image(batch_tensor)
                        else:
                            batch_emb = self.embedding_model(batch_tensor)
                else:
                    if self.embedding_model_name == 'clip':
                        batch_emb = self.embedding_model.encode_image(batch_tensor)
                    else:
                        batch_emb = self.embedding_model(batch_tensor)
                
                batch_emb = batch_emb.cpu().float().numpy()
                
                # Normalize each embedding
                for emb in batch_emb:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    embeddings.append(emb)
        
        return np.array(embeddings)
    
    def extract_embeddings_from_directory(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract embeddings from directory structure: data_dir/modern/ and data_dir/traditional/
        
        Args:
            data_dir: Directory containing modern/ and traditional/ subdirectories
            
        Returns:
            embeddings: Feature matrix (N, embedding_dim)
            labels: Label array (N,)
            image_paths: List of image paths
        """
        embeddings = []
        labels = []
        image_paths = []
        
        data_path = Path(data_dir)
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_path / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            logger.info(f"Processing {len(image_files)} images from {class_name}...")
            
            for img_path in image_files:
                try:
                    embedding = self.extract_embedding(str(img_path))
                    embeddings.append(embedding)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {e}")
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        logger.info(f"✓ Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")
        return embeddings, labels, image_paths
    
    def train(self, X: np.ndarray, y: np.ndarray, use_cross_validation: bool = True):
        """
        Train traditional classifier on embeddings.
        
        Args:
            X: Feature matrix (embeddings)
            y: Labels
            use_cross_validation: Use GridSearchCV for hyperparameter tuning
        """
        logger.info(f"Training {self.classifier_type.upper()} classifier...")
        logger.info(f"Training set: {len(X)} samples, {X.shape[1]} features")
        
        # Standardize features (important for SVM)
        X_scaled = self.scaler.fit_transform(X)
        
        if self.classifier_type == 'svm':
            if use_cross_validation:
                # Grid search for best SVM parameters
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'linear']
                }
                logger.info("Running GridSearchCV for SVM hyperparameters...")
                clf = GridSearchCV(SVC(probability=True, random_state=42), 
                                 param_grid, cv=5, verbose=1, n_jobs=-1)
                clf.fit(X_scaled, y)
                self.classifier = clf.best_estimator_
                logger.info(f"✓ Best SVM params: {clf.best_params_}")
                logger.info(f"✓ Best CV score: {clf.best_score_:.4f}")
            else:
                self.classifier = SVC(kernel='rbf', C=10, gamma='scale', 
                                    probability=True, random_state=42)
                self.classifier.fit(X_scaled, y)
        
        elif self.classifier_type == 'rf':
            if use_cross_validation:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                logger.info("Running GridSearchCV for RandomForest hyperparameters...")
                clf = GridSearchCV(RandomForestClassifier(random_state=42), 
                                 param_grid, cv=5, verbose=1, n_jobs=-1)
                clf.fit(X_scaled, y)
                self.classifier = clf.best_estimator_
                logger.info(f"✓ Best RF params: {clf.best_params_}")
                logger.info(f"✓ Best CV score: {clf.best_score_:.4f}")
            else:
                self.classifier = RandomForestClassifier(n_estimators=100, 
                                                        max_depth=20, 
                                                        random_state=42)
                self.classifier.fit(X_scaled, y)
        
        elif self.classifier_type == 'xgboost':
            if use_cross_validation:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
                logger.info("Running GridSearchCV for GradientBoosting hyperparameters...")
                clf = GridSearchCV(GradientBoostingClassifier(random_state=42), 
                                 param_grid, cv=5, verbose=1, n_jobs=-1)
                clf.fit(X_scaled, y)
                self.classifier = clf.best_estimator_
                logger.info(f"✓ Best XGBoost params: {clf.best_params_}")
                logger.info(f"✓ Best CV score: {clf.best_score_:.4f}")
            else:
                self.classifier = GradientBoostingClassifier(n_estimators=100, 
                                                            max_depth=5, 
                                                            learning_rate=0.1,
                                                            random_state=42)
                self.classifier.fit(X_scaled, y)
        
        # Cross-validation score
        if use_cross_validation:
            cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
            logger.info(f"✓ Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Training accuracy
        train_acc = self.classifier.score(X_scaled, y)
        logger.info(f"✓ Training accuracy: {train_acc:.4f}")
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict facade type for an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            (predicted_class, confidence)
        """
        # Extract embedding
        embedding = self.extract_embedding(image_path)
        embedding = embedding.reshape(1, -1)
        
        # Scale
        embedding_scaled = self.scaler.transform(embedding)
        
        # Predict
        pred = self.classifier.predict(embedding_scaled)[0]
        probs = self.classifier.predict_proba(embedding_scaled)[0]
        confidence = probs[pred]
        
        return self.class_names[pred], confidence
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate classifier on test set.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            metrics dictionary
        """
        X_scaled = self.scaler.transform(X)
        y_pred = self.classifier.predict(X_scaled)
        
        accuracy = (y_pred == y).mean()
        
        logger.info("\n" + "="*60)
        logger.info("Classification Report:")
        logger.info("="*60)
        print(classification_report(y, y_pred, target_names=self.class_names))
        
        logger.info("Confusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def save(self, save_path: str):
        """Save classifier to disk."""
        save_data = {
            'embedding_model_name': self.embedding_model_name,
            'classifier_type': self.classifier_type,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'embedding_dim': self.embedding_dim
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"✓ Model saved to {save_path}")
    
    def load(self, load_path: str):
        """Load classifier from disk."""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.embedding_model_name = save_data['embedding_model_name']
        self.classifier_type = save_data['classifier_type']
        self.classifier = save_data['classifier']
        self.scaler = save_data['scaler']
        self.class_names = save_data['class_names']
        self.embedding_dim = save_data['embedding_dim']
        
        # Reload embedding model
        self._load_embedding_model()
        logger.info(f"✓ Model loaded from {load_path}")


def compare_all_models(data_dir: str, test_split: float = 0.2):
    """
    Compare all embedding + classifier combinations.
    
    Args:
        data_dir: Directory with modern/ and traditional/ subdirectories
        test_split: Fraction to use for testing
    """
    from sklearn.model_selection import train_test_split
    
    embedding_models = ['clip', 'dino', 'vit']
    classifiers = ['svm', 'rf', 'xgboost']
    
    results = []
    
    for emb_model in embedding_models:
        logger.info("\n" + "="*70)
        logger.info(f"TESTING: {emb_model.upper()} Embeddings")
        logger.info("="*70)
        
        try:
            # Create classifier
            clf = EmbeddingClassifier(embedding_model=emb_model, classifier_type='svm')
            
            # Extract embeddings
            X, y, paths = clf.extract_embeddings_from_directory(data_dir)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42, stratify=y
            )
            
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Test each classifier
            for clf_type in classifiers:
                logger.info(f"\n--- {emb_model.upper()} + {clf_type.upper()} ---")
                
                clf.classifier_type = clf_type
                clf.train(X_train, y_train, use_cross_validation=True)
                
                metrics = clf.evaluate(X_test, y_test)
                
                results.append({
                    'embedding': emb_model,
                    'classifier': clf_type,
                    'accuracy': metrics['accuracy'],
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
                
                logger.info(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
        
        except Exception as e:
            logger.error(f"Failed {emb_model}: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY: All Model Comparisons")
    logger.info("="*70)
    logger.info(f"{'Embedding':<15} {'Classifier':<15} {'Accuracy':<10}")
    logger.info("-"*70)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        logger.info(f"{r['embedding']:<15} {r['classifier']:<15} {r['accuracy']:.4f}")
    
    best = max(results, key=lambda x: x['accuracy'])
    logger.info("\n" + "="*70)
    logger.info(f"🏆 BEST: {best['embedding'].upper()} + {best['classifier'].upper()} = {best['accuracy']:.4f}")
    logger.info("="*70)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "facade_images"  # Default directory
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        logger.info("Usage: python embedding_classifier.py <data_directory>")
        logger.info("Expected structure:")
        logger.info("  data_directory/")
        logger.info("    modern/")
        logger.info("      image1.jpg")
        logger.info("      image2.jpg")
        logger.info("    traditional/")
        logger.info("      image1.jpg")
        logger.info("      image2.jpg")
        sys.exit(1)
    
    # Run comparison
    results = compare_all_models(data_dir, test_split=0.2)
