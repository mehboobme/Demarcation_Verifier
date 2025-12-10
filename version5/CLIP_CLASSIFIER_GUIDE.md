# CLIP-Based Automatic Villa Classifier Guide

## Overview

This system uses OpenAI's CLIP model to automatically classify villa features (facade type, render type, view type) based on your 96 reference images. **No API costs** after training!

---

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install PyTorch (choose based on your system)
# CPU only (smaller download):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR GPU (CUDA 11.8 - faster):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

**Download size:** ~2GB (CPU) or ~3GB (GPU)

---

### 2. Organize Your 96 Reference Images

Create this folder structure:

```
reference_images/
    facade_type/
        traditional/
            villa1.jpg
            villa2.jpg
            ...
        modern/
            villa1.jpg
            villa2.jpg
            ...
    render_type/
        white_render/
            villa1.jpg
            ...
        stone/
            villa1.jpg
            ...
        brick/
            villa1.jpg
            ...
    view_type/
        front/
            villa1.jpg
            ...
        rear/
            villa1.jpg
            ...
        side/
            villa1.jpg
            ...
```

**Quick setup:**

```bash
python -c "from villa_classifier import setup_reference_structure; setup_reference_structure()"
```

This creates the folder structure. Then copy your 96 images into the appropriate folders.

---

### 3. Train the Classifier

```bash
python train_classifier.py ./reference_images
```

**What happens:**
- Reads all 96 images
- Creates embeddings (fingerprints) for each image
- Saves trained model to `./models/villa_classifier.pkl`
- **Takes ~5 minutes**

**Output:**
```
============================================================
TRAINING VILLA CLASSIFIER
============================================================

Processing facade_type...
  Category: traditional
    Loaded 40 images
  Category: modern
    Loaded 30 images

Processing render_type...
  Category: white_render
    Loaded 10 images
  Category: stone
    Loaded 8 images

Total: 88 reference images

[OK] Model saved: ./models/villa_classifier.pkl
```

---

### 4. Use in Your Application

The classifier is automatically integrated! Just enable it in your code:

**Option A: In GUI**

```python
# In gui.py (already integrated)
# The GUI will automatically load and use the classifier if available
```

**Option B: In Code**

```python
from villa_classifier import VillaClassifier
from pdf_extractor import PDFExtractor

# Load trained model (once)
classifier = VillaClassifier.load('./models/villa_classifier.pkl')

# Use with PDF extractor
extractor = PDFExtractor(
    pdf_file,
    debug_dir="./debug",
    vision_model="auto",  # Keep as backup
    auto_classifier=classifier  # NEW: Use automatic classifier first!
)

# Extract will automatically use classifier
data = extractor.extract_all()
```

---

## How It Works

### Priority System

When extracting facade type, the system tries in this order:

1. **Automatic Classifier** (70%+ confidence)
   - FREE
   - Fast (~0.5 seconds)
   - Offline
   - Based on your 96 images

2. **Claude Vision API** (if classifier uncertain)
   - $0.01 per image
   - Slow (~3 seconds)
   - Requires internet
   - Backup for difficult cases

3. **Gemini Vision API** (final fallback)
   - $0.005 per image
   - Last resort

### Confidence Threshold

- **≥70% confidence**: Use classifier result
- **<70% confidence**: Fall back to AI Vision

You can adjust this in `pdf_extractor.py line 1471`:
```python
if confidence >= 0.70:  # Change to 0.60, 0.80, etc.
    return
```

---

## Testing the Classifier

### Test on a Single Image

```python
from villa_classifier import VillaClassifier

# Load model
classifier = VillaClassifier.load('./models/villa_classifier.pkl')

# Classify
result = classifier.classify('./test_villa.jpg')

print(result)
# {
#   'facade_type': {
#     'category': 'traditional',
#     'confidence': 0.95,
#     'matches': [
#       {'category': 'traditional', 'similarity': 0.96, 'image': 'ref1.jpg'},
#       {'category': 'traditional', 'similarity': 0.94, 'image': 'ref2.jpg'},
#       ...
#     ]
#   }
# }
```

### Batch Test

Create `test_classifier.py`:
```python
from villa_classifier import VillaClassifier
from pathlib import Path

classifier = VillaClassifier.load('./models/villa_classifier.pkl')

test_images = Path('./test_images').glob('*.jpg')

for img_path in test_images:
    result = classifier.classify(str(img_path))
    facade = result['facade_type']
    print(f"{img_path.name}: {facade['category']} ({facade['confidence']:.1%})")
```

---

## Improving Accuracy

### Add More Reference Images

The more reference images, the better the accuracy!

```bash
# Add new images to your reference folders
reference_images/facade_type/traditional/villa50.jpg

# Retrain (takes ~5 minutes)
python train_classifier.py ./reference_images
```

### Check Misclassifications

```python
# Get top matches to see what it's comparing to
result = classifier.classify('./villa.jpg')

print("Top 5 similar images:")
for match in result['facade_type']['matches'][:5]:
    print(f"  {match['category']}: {match['similarity']:.2%} - {match['image']}")
```

If it's comparing to wrong images, you may need more diverse examples in that category.

---

## Cost Comparison

### Current System (AI Vision Only)

- 100 PDFs × $0.01 = **$1.00 per batch**
- Requires internet
- 3 seconds per image

### With Automatic Classifier

- Training: **One-time (~5 minutes)**
- Classification: **$0.00 per PDF**
- Works offline
- 0.5 seconds per image
- Falls back to AI Vision only when uncertain

**Savings:** ~90% cost reduction (assuming 90% confidence rate)

---

## Troubleshooting

### "CLIP not installed"

```bash
pip install git+https://github.com/openai/CLIP.git
```

### "CUDA out of memory"

Use CPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "No reference images found"

Check folder structure:
```bash
ls reference_images/facade_type/traditional/
```

Should show `.jpg`, `.png`, or `.jpeg` files.

### Low Accuracy

1. **Add more reference images** - Need good coverage of variations
2. **Check image quality** - Blurry/dark images reduce accuracy
3. **Ensure correct categories** - Verify images are in right folders
4. **Lower confidence threshold** - Change from 0.70 to 0.60

---

## Advanced Usage

### Classify Multiple Features

```python
result = classifier.classify(
    './villa.jpg',
    features=['facade_type', 'render_type', 'view_type']
)

print(f"Facade: {result['facade_type']['category']}")
print(f"Render: {result['render_type']['category']}")
print(f"View: {result['view_type']['category']}")
```

### Adjust Number of Neighbors

```python
result = classifier.classify(
    './villa.jpg',
    top_k=10  # Consider top 10 most similar (default: 5)
)
```

### Save Debug Images

```python
# Classifier saves temporary images for classification
# Check: debug_images/<pdf_name>/temp_facade_p3.jpg
```

---

## Next Steps

1. ✅ Install dependencies (Step 1)
2. ✅ Organize 96 reference images (Step 2)
3. ✅ Train classifier (Step 3)
4. ✅ Test on sample PDFs
5. ✅ Monitor accuracy and adjust

**Your system is now fully automatic!** 🎉

For questions or issues, check the logs in the PDF extraction output.
