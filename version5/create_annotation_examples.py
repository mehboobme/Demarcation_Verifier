"""
Annotation Example Visualizer
------------------------------
Shows sample annotations overlaid on images to guide annotation process.
"""

import cv2
import numpy as np
from pathlib import Path

# Annotation examples for reference
ANNOTATION_EXAMPLES = {
    "DM1-D02-2A-25-453-01_page1.png": {
        "plot": [(800, 900, 1200, 1300)],  # x1, y1, x2, y2 (approximate)
        "plot_number": [(950, 1050, 1050, 1150)],  # "453"
        "unit_type": [(950, 900, 1050, 1000)],  # Type label
        "adjacent_plot": [(600, 900, 800, 1300), (1200, 900, 1400, 1300)],  # N-1, N+1
        "street": [(700, 800, 1500, 850)],  # Top street
        "park": []  # No park adjacent
    },
    "DM1-D02-2A-26-476-01_page1.png": {
        "plot": [(800, 1000, 1200, 1400)],
        "plot_number": [(950, 1150, 1050, 1250)],
        "unit_type": [(950, 1000, 1050, 1100)],
        "adjacent_plot": [(600, 1000, 800, 1400)],  # Only left neighbor (villa)
        "street": [(700, 900, 1500, 950), (800, 1400, 1200, 1450)],
        "park": [(1200, 1000, 1600, 1400)]  # Right side is park (CORNER)
    }
}

# Colors for each class (BGR format for OpenCV)
COLORS = {
    "plot": (0, 0, 255),           # Red
    "plot_number": (0, 255, 0),    # Green
    "unit_type": (255, 0, 0),      # Blue
    "street": (0, 255, 255),       # Yellow
    "park": (255, 255, 0),         # Cyan
    "adjacent_plot": (255, 0, 255) # Magenta
}

def visualize_example(image_name):
    """Create example annotation visualization"""
    img_path = Path(__file__).parent / "annotation_images" / image_name
    
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        return
    
    # Get example annotations
    annotations = ANNOTATION_EXAMPLES.get(image_name, {})
    
    # Draw bounding boxes
    for class_name, boxes in annotations.items():
        color = COLORS[class_name]
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            label = class_name.upper()
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add instructions
    cv2.putText(img, "EXAMPLE ANNOTATION GUIDE", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Save
    output_path = Path(__file__).parent / f"example_annotation_{image_name}"
    cv2.imwrite(str(output_path), img)
    print(f"✓ Created example: {output_path}")
    
    return img

def main():
    print("=" * 70)
    print("ANNOTATION EXAMPLE GENERATOR")
    print("=" * 70)
    print("\nThis creates visual examples showing how to annotate.")
    print("Use these as reference when annotating in Label Studio.\n")
    
    for img_name in ANNOTATION_EXAMPLES.keys():
        visualize_example(img_name)
    
    print("\n" + "=" * 70)
    print("CLASS DESCRIPTIONS:")
    print("=" * 70)
    print("\n🔴 plot (Red):")
    print("   - The main highlighted villa plot (usually red border)")
    print("   - Draw tight box around entire plot boundary")
    
    print("\n🟢 plot_number (Green):")
    print("   - Numbers like 453, 476, 526")
    print("   - Include ALL visible plot numbers (target + neighbors)")
    
    print("\n🔵 unit_type (Blue):")
    print("   - Type labels: V2C, DA2, V3A, etc.")
    print("   - Usually inside plot or near boundary")
    
    print("\n🟡 street (Yellow):")
    print("   - Street grid lines and labels")
    print("   - Include width markers (6m, 8m, 10m)")
    print("   - Exclude corridors (ممر)")
    
    print("\n🔵 park (Cyan):")
    print("   - Parks and open spaces")
    print("   - Grid patterns WITHOUT plot numbers")
    
    print("\n🟣 adjacent_plot (Magenta):")
    print("   - Plots immediately next to red highlighted plot")
    print("   - Left (N-1) and right (N+1) neighbors")
    print("   - Helps determine corner status")
    
    print("\n" + "=" * 70)
    print("\nNOTE: These examples show APPROXIMATE bounding boxes.")
    print("In Label Studio, draw precise boxes around actual elements.")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
