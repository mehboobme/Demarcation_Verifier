"""
Label Studio Annotation Launcher
---------------------------------
This script initializes and launches Label Studio for annotating plot detection images.

Usage:
    python start_annotation.py

Steps:
1. Initializes Label Studio project
2. Imports images from annotation_images/ folder
3. Opens Label Studio web interface at http://localhost:8080
4. Login credentials: admin / password (you can change this)
"""

import os
import json
import subprocess
import webbrowser
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
ANNOTATION_IMAGES_DIR = BASE_DIR / "annotation_images"
LABEL_STUDIO_DIR = BASE_DIR / "label_studio_data"
CONFIG_FILE = BASE_DIR / "label_studio_config.xml"

# Label Studio configuration for object detection
LABEL_STUDIO_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="plot" background="#FF0000"/>
    <Label value="plot_number" background="#00FF00"/>
    <Label value="unit_type" background="#0000FF"/>
    <Label value="street" background="#FFFF00"/>
    <Label value="park" background="#00FFFF"/>
    <Label value="adjacent_plot" background="#FF00FF"/>
  </RectangleLabels>
</View>"""

def create_label_studio_config():
    """Create Label Studio labeling configuration"""
    print("📝 Creating Label Studio configuration...")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(LABEL_STUDIO_CONFIG)
    print(f"✓ Configuration saved to: {CONFIG_FILE}")

def create_import_json():
    """Create JSON file with image paths for Label Studio import"""
    print("\n📸 Preparing images for import...")
    
    images = list(ANNOTATION_IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"❌ No PNG images found in {ANNOTATION_IMAGES_DIR}")
        return None
    
    # Create tasks JSON for import
    tasks = []
    for img in images:
        # Use absolute path for Label Studio
        img_path = str(img.absolute()).replace('\\', '/')
        tasks.append({
            "data": {
                "image": img_path
            }
        })
    
    import_file = BASE_DIR / "label_studio_import.json"
    with open(import_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"✓ Prepared {len(tasks)} images for annotation")
    print(f"✓ Import file: {import_file}")
    return import_file

def main():
    print("=" * 70)
    print("LABEL STUDIO - YOLO PLOT DETECTION ANNOTATION")
    print("=" * 70)
    
    # Check if images exist
    if not ANNOTATION_IMAGES_DIR.exists():
        print(f"\n❌ ERROR: {ANNOTATION_IMAGES_DIR} not found!")
        print("Run: python train_yolo_plot_detector.py")
        return
    
    # Create configuration
    create_label_studio_config()
    
    # Create import JSON
    import_file = create_import_json()
    if not import_file:
        return
    
    print("\n" + "=" * 70)
    print("STARTING LABEL STUDIO")
    print("=" * 70)
    print("\n📌 IMPORTANT INSTRUCTIONS:")
    print("\n1. Label Studio will open at: http://localhost:8080")
    print("2. First time setup:")
    print("   - Create account with username and password")
    print("   - Or skip and use default admin/password")
    print("\n3. Create a new project:")
    print("   - Name: 'YOLO Plot Detection'")
    print("   - Go to Settings > Labeling Interface")
    print("   - Click 'Code' and paste the XML from:", CONFIG_FILE)
    print("\n4. Import images:")
    print("   - Go to project > Import")
    print("   - Upload:", import_file)
    print("   - Or drag-drop PNG files from:", ANNOTATION_IMAGES_DIR)
    print("\n5. Annotate 6 classes:")
    print("   🔴 plot - Red highlighted villa plots")
    print("   🟢 plot_number - Numbers (453, 476, 526, etc.)")
    print("   🔵 unit_type - Type labels (V2C, DA2, etc.)")
    print("   🟡 street - Street grid lines")
    print("   🔵 park - Park/open spaces")
    print("   🟣 adjacent_plot - Neighboring plots")
    print("\n6. Export annotations:")
    print("   - After annotation complete, go to Export")
    print("   - Select: YOLO format")
    print("   - Download and extract to: yolo_dataset/")
    print("\n7. Then run training:")
    print("   - python train_yolo_plot_detector.py --train")
    print("\n" + "=" * 70)
    
    # Launch Label Studio
    print("\n🚀 Launching Label Studio...")
    print("Press Ctrl+C to stop the server when done.\n")
    
    try:
        # Initialize Label Studio (creates database)
        subprocess.run([
            "label-studio", "init", str(LABEL_STUDIO_DIR)
        ], check=False)
        
        # Start Label Studio server
        print("\n🌐 Opening browser in 3 seconds...")
        time.sleep(3)
        webbrowser.open("http://localhost:8080")
        
        # Run Label Studio
        subprocess.run([
            "label-studio", "start", str(LABEL_STUDIO_DIR),
            "--port", "8080"
        ])
        
    except KeyboardInterrupt:
        print("\n\n✓ Label Studio stopped.")
        print("Run this script again anytime to resume annotation.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running manually:")
        print("label-studio start")

if __name__ == "__main__":
    main()
