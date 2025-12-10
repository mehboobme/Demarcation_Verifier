# YOLO Plot Detection Training Pipeline

## Overview

This pipeline trains a custom YOLOv8 model to accurately detect and classify elements in ROSHN demarcation PDFs, enabling 99%+ accuracy in corner plot detection.

## Current Status

✅ **Installed**: YOLOv8, Roboflow, Supervision, PaddleOCR, Tesseract OCR  
✅ **Prepared**: 12 PDF pages converted to high-res images (200 DPI)  
⏳ **Next Step**: Annotate images with bounding boxes

## What YOLO Will Detect

The model will be trained to detect 6 classes:

1. **plot** - Red highlighted target plot boundaries
2. **plot_number** - Plot number text (e.g., "453", "476", "526")
3. **unit_type** - Unit type labels (e.g., "V2C", "DA2", "DB3")
4. **street** - Street/road areas (bright grid patterns)
5. **park** - Park/open space areas (grid patterns without plots)
6. **adjacent_plot** - Neighboring villa plots (non-highlighted)

## Why This Improves Accuracy

**Current approach (97.4% accuracy):**
- OCR (Tesseract/PaddleOCR) struggles with angled text and varying orientations
- Visual heuristics (fill ratios, contours) can confuse parks with villa plots
- Multi-region search is complex and error-prone

**YOLO approach (expected 99%+ accuracy):**
- Direct object detection trained on actual PDF layouts
- Learns spatial relationships between plots, numbers, and streets
- Handles angled/trapezoid plots naturally
- Single inference provides all detections simultaneously
- GPU-accelerated for fast processing

## Annotation Process

### Option 1: Roboflow (Recommended - Easier)

1. **Sign up**: https://roboflow.com (free tier)
2. **Create project**: 
   - Project name: "ROSHN Plot Detection"
   - Project type: Object Detection
   - Annotation group: Choose "Bounding Box"
3. **Upload images**: 
   - Upload all 12 images from `annotation_images/`
4. **Annotate**:
   - Draw bounding boxes around each element
   - Classes:
     - `plot`: Red highlighted plot
     - `plot_number`: Numbers inside plots
     - `unit_type`: Type labels (V2C, DA2, etc.)
     - `street`: Street grid patterns
     - `park`: Park/open areas
     - `adjacent_plot`: Neighboring plots
5. **Split dataset**:
   - 10 images → Training
   - 2 images → Validation
6. **Export**:
   - Format: YOLOv8
   - Download and extract to `yolo_dataset/`

### Option 2: Label Studio (Open Source)

```bash
pip install label-studio
label-studio
```

1. Open http://localhost:8080
2. Create new project → Object Detection
3. Import images from `annotation_images/`
4. Configure bounding box labeling
5. Annotate all 12 images
6. Export in YOLO format
7. Place in `yolo_dataset/`

### Annotation Tips

- **plot**: Draw tight box around the red highlighted area
- **plot_number**: Box around the number text only (453, 476, etc.)
- **unit_type**: Box around type text (V2C, DA2, etc.)
- **street**: Boxaround street grid patterns (bright, with grid lines)
- **park**: Box around park areas (grid but no plot numbers)
- **adjacent_plot**: Box around nearby villa plots (not highlighted)

**Time estimate**: ~10-15 minutes per image = 2-3 hours total

## Training the Model

Once annotations are complete:

```bash
python train_yolo_plot_detector.py --train
```

**Training parameters:**
- Model: YOLOv8m (medium - good balance)
- Epochs: 100
- Image size: 1024x1024 (high resolution)
- Batch size: 4 (adjust based on GPU memory)
- Early stopping: Patience 20 epochs

**Training time**: ~30-60 minutes on RTX 5070

## Using the Trained Model

After training, the model will automatically integrate:

```python
from yolo_corner_detector import YOLOCornerDetector

# Initialize detector
detector = YOLOCornerDetector('plot_detection_training/yolo_plot_detector/weights/best.pt')

# Detect corner
is_corner, connected_streets, details = detector.detect_corner(image, plot_number=476)

print(f"Plot 476 is corner: {is_corner}")
print(f"Connected streets: {connected_streets}")
print(f"Details: {details}")
```

## Expected Results

**Before YOLO (current):**
- Accuracy: 97.4%
- Mismatches: 13/492 checks
- Issues: Angled plots, parks vs villas confusion

**After YOLO (trained):**
- Expected accuracy: 99%+
- Mismatches: <5/492 checks
- Benefits: Direct detection, handles all orientations

## Model Performance Metrics

After training, check:

```bash
# View training results
tensorboard --logdir plot_detection_training/yolo_plot_detector

# Test on validation set
python -c "
from ultralytics import YOLO
model = YOLO('plot_detection_training/yolo_plot_detector/weights/best.pt')
model.val()
"
```

## Integration with pdf_extractor.py

The YOLO detector will be automatically used if the model exists. It falls back to OCR + visual heuristics if not trained yet.

Priority order:
1. **YOLO** (if model exists) - Best accuracy
2. **PaddleOCR** - Better than Tesseract
3. **Tesseract OCR** - Fallback
4. **Visual heuristics** - Last resort

## File Structure

```
Demarcation_Verifier/version5/
├── train_yolo_plot_detector.py    # Training pipeline
├── yolo_corner_detector.py        # YOLO integration
├── annotation_images/              # Images ready for annotation
│   ├── DM1-D02-2A-25-453-01_page1.png
│   ├── DM1-D02-2A-26-476-01_page1.png
│   └── ... (12 images total)
├── yolo_dataset/                   # Annotated dataset (after annotation)
│   ├── images/
│   │   ├── train/                  # 10 training images
│   │   └── val/                    # 2 validation images
│   ├── labels/
│   │   ├── train/                  # 10 training labels (.txt)
│   │   └── val/                    # 2 validation labels (.txt)
│   └── dataset.yaml                # Auto-generated config
└── plot_detection_training/        # Training outputs (after training)
    └── yolo_plot_detector/
        ├── weights/
        │   ├── best.pt             # Best model weights
        │   └── last.pt             # Last epoch weights
        ├── results.png             # Training curves
        └── confusion_matrix.png    # Confusion matrix
```

## Next Steps

1. **Annotate 12 images** using Roboflow or Label Studio (2-3 hours)
2. **Run training**: `python train_yolo_plot_detector.py --train` (30-60 min)
3. **Validate model**: Check accuracy on validation set
4. **Test on all PDFs**: Run full validation to confirm 99%+ accuracy
5. **Deploy**: Model automatically integrates with pdf_extractor.py

## Questions?

- Training issues: Check GPU memory, reduce batch size
- Annotation questions: See example images in `annotation_images/`
- Model not loading: Verify path to `best.pt` weights file

---

**Current Status**: Ready for annotation! 📝  
**Next Action**: Annotate 12 images in Roboflow or Label Studio
