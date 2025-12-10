# CLIP Training Quick Start Guide

## Step 1: Verify Your Reference Images

Your reference images should be organized as:

```
reference_images/
├── facade_types/
│   ├── traditional/
│   │   ├── traditional_front_elevation_01.jpg
│   │   ├── traditional_rear_elevation_01.jpg
│   │   └── ... (more traditional facade images)
│   └── modern/
│       ├── modern_front_elevation_01.jpg
│       ├── modern_side_elevation_01.jpg
│       └── ... (more modern facade images)
└── floor_plans/
    ├── traditional/
    │   ├── traditional_ground_floor_01.jpg
    │   └── ... (more traditional floor plans)
    └── modern/
        ├── modern_ground_floor_01.jpg
        └── ... (more modern floor plans)
```

## Step 2: Install CLIP (if not already installed)

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Step 3: Train the Model

### Option A: Default Settings (Recommended)
```bash
python train_clip_model.py
```

This will:
- Look for images in `./reference_images/`
- Train on all facade_types and floor_plans
- Save model to `./models/roshn_classifier.pkl`

### Option B: Custom Settings
```bash
python train_clip_model.py \
    --reference-dir ./my_references \
    --output ./models/my_classifier.pkl \
    --model ViT-B/16
```

Available CLIP models:
- `ViT-B/32` - Fast, good accuracy (default, ~350MB)
- `ViT-B/16` - Better accuracy, slower (~350MB)
- `ViT-L/14` - Best accuracy, slowest (~900MB)
- `RN50` - ResNet-based (~400MB)

## Step 4: Test the Model

Test on a single image:
```bash
python test_classifier.py ./models/roshn_classifier.pkl ./test_image.jpg
```

Expected output:
```
FACADE TYPE:
  Predicted: traditional
  Confidence: 94.5%
  Votes: {'traditional': 5, 'modern': 0}
  
  Top 5 matches:
    1. traditional (0.945) - traditional_front_elevation_02.jpg
    2. traditional (0.932) - traditional_rear_elevation_01.jpg
    ...
```

## Step 5: Integrate with Your System

### In pdf_extractor.py

The model is automatically used if available. Just ensure the model file exists:

```python
# The system will automatically try to load:
# ./models/roshn_classifier.pkl

# When processing a PDF, it will use CLIP first, then fallback to AI APIs
```

### Manual Usage

```python
from train_clip_model import ROSHNClassifier

# Load trained model
classifier = ROSHNClassifier.load('./models/roshn_classifier.pkl')

# Classify an image
result = classifier.classify('./villa_page.jpg')

# Get facade type
facade_result = result.get('facade_type', {})
print(f"Facade: {facade_result['category']}")
print(f"Confidence: {facade_result['confidence']:.2%}")

# Get floor plan type
plan_result = result.get('floor_plan_type', {})
print(f"Floor Plan: {plan_result['category']}")
```

## Tips for Better Accuracy

1. **More Images**: 10-15 images per category is ideal
2. **Good Quality**: Use high-resolution images (200+ DPI)
3. **Clear Headers**: Filenames with descriptive names help debugging
4. **Consistent Views**: Use similar page types (e.g., all front elevations)
5. **Diverse Examples**: Include variations within each category

## Troubleshooting

### Issue: "No images found"
- Check image file extensions (.jpg, .jpeg, .png)
- Verify directory structure matches expected layout
- Ensure images are in category subfolders, not root

### Issue: "Low confidence"
- Add more reference images
- Use better quality source images
- Try a larger CLIP model (ViT-B/16 or ViT-L/14)

### Issue: "CUDA out of memory"
- Use smaller model (ViT-B/32 instead of ViT-L/14)
- Process images one at a time
- Reduce image resolution in preprocessing

### Issue: Model file too large
- ViT-B/32 model creates ~50-100MB .pkl files
- This is normal - embeddings are pre-computed for fast inference
- Consider using fewer reference images if storage is limited

## Performance Benchmarks

| CLIP Model | Training Time | Inference Time | Accuracy |
|------------|--------------|----------------|----------|
| ViT-B/32   | ~1-2 min     | ~100ms/image   | Good     |
| ViT-B/16   | ~2-3 min     | ~150ms/image   | Better   |
| ViT-L/14   | ~5-10 min    | ~300ms/image   | Best     |

*Times based on 50 reference images on CPU*

## Next Steps

After training:
1. Test on sample PDFs
2. Monitor accuracy in validation reports
3. Add more reference images if needed
4. Retrain periodically with new examples
