"""
YOLO Plot Detection Training Pipeline
Trains a custom YOLOv8 model to detect:
1. Plot boundaries
2. Plot numbers
3. Street areas
4. Park areas
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
import fitz  # PyMuPDF
from PIL import Image
import json

class PlotDetectorTrainer:
    """Train YOLO model for plot detection in PDF demarcation drawings"""
    
    def __init__(self, dataset_dir='yolo_dataset'):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # Create dataset structure
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Class definitions
        self.classes = {
            0: 'plot',           # Villa plot boundary (red highlighted area)
            1: 'plot_number',    # Plot number text (e.g., "453", "476")
            2: 'unit_type',      # Unit type text (e.g., "V2C", "DA2")
            3: 'street',         # Street/road area (grid pattern, bright)
            4: 'park',           # Park/open space (grid pattern with no plot)
            5: 'adjacent_plot'   # Non-highlighted adjacent plot
        }
        
        print(f"✓ Dataset directory initialized: {self.dataset_dir}")
        print(f"  Classes: {self.classes}")
    
    def prepare_pdfs_for_annotation(self, pdf_folder='input_data', output_folder='annotation_images'):
        """
        Convert PDFs to images for annotation
        Extracts the plot layout pages (typically page 1)
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        pdf_files = list(Path(pdf_folder).glob('*.pdf'))
        print(f"\n📄 Found {len(pdf_files)} PDFs to prepare for annotation")
        
        for pdf_file in pdf_files:
            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(str(pdf_file))
                
                # Convert first page to image (200 DPI)
                page = doc[0]
                mat = fitz.Matrix(200/72, 200/72)  # 200 DPI scaling
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                output_filename = output_path / f"{pdf_file.stem}_page1.png"
                pix.save(output_filename)
                
                doc.close()
                print(f"  ✓ {pdf_file.name} → {output_filename.name}")
            
            except Exception as e:
                print(f"  ✗ Error processing {pdf_file.name}: {e}")
        
        print(f"\n✓ Images saved to: {output_folder}")
        print("\n📝 Next steps:")
        print("  1. Use Label Studio, Roboflow, or CVAT to annotate these images")
        print("  2. Label the following:")
        print("     - plot: Red highlighted plot boundaries")
        print("     - plot_number: Numbers inside plots (453, 476, etc.)")
        print("     - unit_type: Unit type labels (V2C, DA2, etc.)")
        print("     - street: Street/road grid patterns")
        print("     - park: Park/open space areas")
        print("     - adjacent_plot: Neighboring plots")
        print(f"  3. Export annotations in YOLO format")
        print(f"  4. Place in {self.dataset_dir}/")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        yaml_content = f"""# Plot Detection Dataset
path: {self.dataset_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: plot
  1: plot_number
  2: unit_type
  3: street
  4: park
  5: adjacent_plot

# Number of classes
nc: {len(self.classes)}
"""
        yaml_path = self.dataset_dir / 'dataset.yaml'
        yaml_path.write_text(yaml_content)
        print(f"✓ Created dataset.yaml: {yaml_path}")
        return yaml_path
    
    def train_model(self, epochs=100, img_size=1024, batch_size=8, model_size='n'):
        """
        Train YOLOv8 model
        
        Args:
            epochs: Number of training epochs
            img_size: Input image size (1024 for high-res PDF pages)
            batch_size: Batch size for training
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        # Create dataset yaml
        dataset_yaml = self.create_dataset_yaml()
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
        
        print(f"\n🚀 Starting training...")
        print(f"  Model: YOLOv8{model_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {img_size}")
        print(f"  Batch size: {batch_size}")
        
        # Train model
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=0,  # GPU
            project='plot_detection_training',
            name='yolo_plot_detector',
            exist_ok=True,
            patience=20,  # Early stopping
            save=True,
            plots=True,
            val=True,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
        )
        
        print(f"\n✓ Training complete!")
        print(f"  Best model: plot_detection_training/yolo_plot_detector/weights/best.pt")
        
        return model
    
    def test_model(self, model_path, test_image_path):
        """Test trained model on a sample image"""
        model = YOLO(model_path)
        
        # Run inference
        results = model(test_image_path, conf=0.25)
        
        # Visualize results
        for r in results:
            im_array = r.plot()  # Plot with bounding boxes
            cv2.imshow('Detection Results', im_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results


def main():
    """Main training pipeline"""
    trainer = PlotDetectorTrainer()
    
    # Step 1: Prepare PDFs for annotation
    print("=" * 60)
    print("STEP 1: Prepare Images for Annotation")
    print("=" * 60)
    trainer.prepare_pdfs_for_annotation()
    
    print("\n" + "=" * 60)
    print("ANNOTATION INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Use Label Studio (free, open-source):
   - Install: pip install label-studio
   - Run: label-studio
   - Open browser at http://localhost:8080
   - Create new project
   - Import images from annotation_images/
   - Use bounding box tool to label:
     * plot: Red highlighted plot area
     * plot_number: Number text (453, 476, etc.)
     * unit_type: Type labels (V2C, DA2, etc.)
     * street: Street/road areas
     * park: Park/open spaces
     * adjacent_plot: Neighboring plots
   - Export in YOLO format
   
2. Or use Roboflow (easier, cloud-based):
   - Sign up at https://roboflow.com (free tier)
   - Create new project → Object Detection
   - Upload images
   - Annotate using their web interface
   - Export as YOLOv8 format
   - Download and extract to yolo_dataset/

3. Place annotations:
   - Training images: yolo_dataset/images/train/
   - Training labels: yolo_dataset/labels/train/
   - Validation images: yolo_dataset/images/val/
   - Validation labels: yolo_dataset/labels/val/
   
   Split: 80% train, 20% validation (e.g., 10 PDFs train, 2 PDFs val)

After annotation, run training with:
    python train_yolo_plot_detector.py --train
""")
    
    # Check if user wants to train
    import sys
    if '--train' in sys.argv:
        print("\n" + "=" * 60)
        print("STEP 2: Training YOLO Model")
        print("=" * 60)
        
        # Check if dataset exists
        if not (trainer.images_dir / 'train').exists() or \
           not list((trainer.images_dir / 'train').glob('*')):
            print("❌ No training images found!")
            print("   Please annotate images first and place them in yolo_dataset/")
            return
        
        # Train model
        model = trainer.train_model(
            epochs=100,
            img_size=1024,
            batch_size=4,  # Adjust based on GPU memory
            model_size='m'  # Medium model (good balance)
        )
        
        print("\n✓ Training pipeline complete!")
        print("\nTo use the trained model:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('plot_detection_training/yolo_plot_detector/weights/best.pt')")
        print("  results = model('test_image.png')")


if __name__ == '__main__':
    main()
