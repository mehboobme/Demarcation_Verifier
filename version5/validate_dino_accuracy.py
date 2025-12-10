"""
Validate DINO + SVM classifier accuracy on known facade images
Uses the debug_images from previous validation runs
"""

import os
from pathlib import Path
import logging
from embedding_classifier import EmbeddingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known facade types from your ground truth
GROUND_TRUTH_FACADE = {
    # Modern facades
    'DM1-D02-2A-25-453-01': 'M',
    'DM1-D02-2A-25-457-01': 'M',
    'DM1-D02-2A-25-462-01': 'M',
    'DM1-D02-2A-26-476-01': 'M',
    'DM1-D02-2A-27-499-00': 'M',
    'DM1-D02-2A-31-554-01': 'M',
    'DM1-D02-2B-32-578-01': 'M',
    'DM1-D02-2C-45-779-01': 'M',
    
    # Traditional facades  
    'DM1-D02-2A-29-526-01': 'T',
    'DM1-D02-2A-31-566-01': 'T',
    'DM1-D02-2B-43-740-01': 'T',
    'DM1-D02-2B-43-741-01': 'T',
}

def test_on_debug_images():
    """Test DINO + SVM on facade images from debug_images directory."""
    
    logger.info("="*70)
    logger.info("DINO + SVM Validation on Debug Images")
    logger.info("="*70)
    
    # Load classifier
    model_path = "dino_svm_facade_classifier.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please run: python train_dino_svm.py")
        return
    
    clf = EmbeddingClassifier(embedding_model='dino', classifier_type='svm')
    clf.load(model_path)
    logger.info(f"✓ Loaded model: {model_path}\n")
    
    # Test on debug images
    debug_dir = Path("debug_images")
    if not debug_dir.exists():
        logger.error("debug_images/ directory not found!")
        logger.info("Please run validation on some PDFs first to generate debug images.")
        return
    
    results = []
    correct = 0
    total = 0
    
    for unit_code, expected_facade in GROUND_TRUTH_FACADE.items():
        unit_dir = debug_dir / unit_code
        
        if not unit_dir.exists():
            continue
        
        # Find facade image
        facade_images = list(unit_dir.glob("*facade*.jpg")) + list(unit_dir.glob("*facade*.png"))
        
        if not facade_images:
            logger.warning(f"No facade image found for {unit_code}")
            continue
        
        # Test on first facade image
        facade_path = str(facade_images[0])
        predicted_class, confidence = clf.predict(facade_path)
        
        # Map to T/M
        facade_map = {'traditional': 'T', 'modern': 'M'}
        predicted = facade_map[predicted_class]
        
        total += 1
        is_correct = (predicted == expected_facade)
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        logger.info(f"{status} {unit_code}: Expected {expected_facade}, Got {predicted} "
                   f"({confidence:.1%} confidence)")
        
        results.append({
            'unit_code': unit_code,
            'expected': expected_facade,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Summary
    if total > 0:
        accuracy = correct / total
        logger.info("\n" + "="*70)
        logger.info("VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"Total tested: {total}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.1%}")
        logger.info("="*70)
        
        # Show mismatches
        mismatches = [r for r in results if not r['correct']]
        if mismatches:
            logger.info("\nMismatches:")
            for m in mismatches:
                logger.info(f"  {m['unit_code']}: Expected {m['expected']}, "
                           f"Got {m['predicted']} ({m['confidence']:.1%})")
        else:
            logger.info("\n🎉 Perfect accuracy! No mismatches found.")
    else:
        logger.warning("No test images found in debug_images/")
        logger.info("\nTo generate test images:")
        logger.info("  1. Run validation on some PDFs")
        logger.info("  2. Facade images will be saved to debug_images/<unit_code>/")

if __name__ == "__main__":
    test_on_debug_images()
