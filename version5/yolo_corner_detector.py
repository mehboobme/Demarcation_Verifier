"""
YOLO-based Corner Plot Detection
Uses trained YOLO model to detect plots and determine corner status
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install with: pip install ultralytics")


class YOLOCornerDetector:
    """Detect corner plots using YOLO object detection"""
    
    def __init__(self, model_path='plot_detection_training/yolo_plot_detector/weights/best.pt'):
        """
        Initialize YOLO corner detector
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model = None
        self.model_path = model_path
        
        if YOLO_AVAILABLE and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                logger.info(f"✓ YOLO plot detector loaded: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
        else:
            logger.info("YOLO model not found. Train model first with train_yolo_plot_detector.py")
    
    def detect_corner(self, image: np.ndarray, target_plot_number: int) -> Tuple[bool, int, dict]:
        """
        Detect if a plot is a corner plot using YOLO
        
        Args:
            image: Input image (plot layout page)
            target_plot_number: The plot number we're analyzing
        
        Returns:
            (is_corner, connected_streets, details)
        """
        if self.model is None:
            logger.warning("YOLO model not loaded, using fallback")
            return None, None, {}
        
        # Run YOLO detection
        results = self.model(image, conf=0.3, verbose=False)
        
        if not results or len(results) == 0:
            return None, None, {}
        
        result = results[0]
        boxes = result.boxes
        
        # Parse detections
        plot_boxes = []
        plot_numbers = []
        streets = []
        parks = []
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if cls == 0:  # plot
                plot_boxes.append({'box': xyxy, 'conf': conf})
            elif cls == 1:  # plot_number
                # OCR the number from this region
                x1, y1, x2, y2 = map(int, xyxy)
                number_region = image[y1:y2, x1:x2]
                number = self._extract_number(number_region)
                if number:
                    plot_numbers.append({'number': number, 'box': xyxy, 'conf': conf})
            elif cls == 3:  # street
                streets.append({'box': xyxy, 'conf': conf})
            elif cls == 4:  # park
                parks.append({'box': xyxy, 'conf': conf})
        
        # Find target plot
        target_plot = None
        for plot_num in plot_numbers:
            if plot_num['number'] == target_plot_number:
                target_plot = plot_num
                break
        
        if not target_plot:
            logger.warning(f"Target plot {target_plot_number} not detected")
            return None, None, {}
        
        # Check for adjacent plots (N-1 and N+1)
        adjacent_nums = [target_plot_number - 1, target_plot_number + 1]
        found_adjacent = []
        
        for plot_num in plot_numbers:
            if plot_num['number'] in adjacent_nums:
                found_adjacent.append(plot_num['number'])
        
        # Determine corner status
        # Corner if missing one or both adjacent plots
        is_corner = len(found_adjacent) < 2
        
        # Count streets adjacent to target plot
        target_box = target_plot['box']
        connected_streets = self._count_adjacent_streets(target_box, streets)
        
        details = {
            'target_plot': target_plot,
            'adjacent_plots_found': found_adjacent,
            'total_plots_detected': len(plot_numbers),
            'streets_detected': len(streets),
            'parks_detected': len(parks),
            'detection_confidence': target_plot['conf']
        }
        
        logger.info(f"YOLO Detection: plot={target_plot_number}, adjacent={found_adjacent}, corner={is_corner}")
        
        return is_corner, connected_streets, details
    
    def _extract_number(self, region: np.ndarray) -> Optional[int]:
        """Extract plot number from region using OCR"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(region, config='--psm 7 digits')
            # Extract first number
            import re
            numbers = re.findall(r'\d{2,4}', text)
            if numbers:
                return int(numbers[0])
        except:
            pass
        return None
    
    def _count_adjacent_streets(self, plot_box: np.ndarray, streets: List[dict]) -> int:
        """Count streets adjacent to plot"""
        if not streets:
            return 1  # Default minimum
        
        # Check IoU with street boxes
        adjacent_count = 0
        px1, py1, px2, py2 = plot_box
        
        for street in streets:
            sx1, sy1, sx2, sy2 = street['box']
            
            # Check if street is adjacent (touching but not overlapping much)
            # Expand plot box slightly to detect adjacency
            margin = 50
            if (sx1 < px2 + margin and sx2 > px1 - margin and
                sy1 < py2 + margin and sy2 > py1 - margin):
                adjacent_count += 1
        
        return max(1, min(adjacent_count, 2))  # Clamp to [1, 2]
    
    def visualize_detections(self, image: np.ndarray, save_path: Optional[str] = None):
        """Visualize YOLO detections on image"""
        if self.model is None:
            return None
        
        results = self.model(image, conf=0.3)
        annotated = results[0].plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated)
        
        return annotated


# Integration with existing pdf_extractor
def use_yolo_for_corner_detection(image: np.ndarray, plot_number: int) -> Tuple[Optional[bool], Optional[int]]:
    """
    Wrapper function to integrate YOLO corner detection
    
    Returns:
        (is_corner, connected_streets) or (None, None) if YOLO unavailable
    """
    detector = YOLOCornerDetector()
    
    if detector.model is None:
        return None, None
    
    is_corner, streets, details = detector.detect_corner(image, plot_number)
    
    return is_corner, streets
