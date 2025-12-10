"""
Train and save DINO + SVM classifier for production use
"""

from embedding_classifier import EmbeddingClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    """Train DINO + SVM on facade dataset and save for production."""
    
    logger.info("="*70)
    logger.info("Training DINO + SVM Facade Classifier (OPTIMIZED)")
    logger.info("="*70)
    
    # Initialize DINO + SVM with optimizations
    clf = EmbeddingClassifier(
        embedding_model='dino', 
        classifier_type='svm', 
        device='cuda',
        use_fp16=True,      # Enable FP16 for faster inference
        batch_size=8        # Process 8 images at once
    )
    
    # Extract embeddings from reference images
    data_dir = r"reference_images\facade_types"
    logger.info(f"\nExtracting embeddings from: {data_dir}")
    X, y, paths = clf.extract_embeddings_from_directory(data_dir)
    
    logger.info(f"Dataset: {len(X)} images ({(y==0).sum()} modern, {(y==1).sum()} traditional)")
    
    # Train on ALL data (no split, since we'll validate on actual PDFs)
    logger.info("\nTraining on full dataset...")
    clf.train(X, y, use_cross_validation=True)
    
    # Save model
    model_path = "dino_svm_facade_classifier.pkl"
    clf.save(model_path)
    
    logger.info("\n" + "="*70)
    logger.info(f"✓ Model trained and saved: {model_path}")
    logger.info("  Optimizations: FP16 + Batch size 8")
    logger.info("="*70)
    
    return clf

if __name__ == "__main__":
    train_and_save_model()
